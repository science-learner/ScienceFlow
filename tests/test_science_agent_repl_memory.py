"""Science Agent Repl contracts: memory."""

from __future__ import annotations

from tests._science_agent_repl_support import *  # noqa: F401,F403


@pytest.mark.asyncio
async def test_long_horizon_compact_inband_drops_recent_tail(tmp_path: Path) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(Message.assistant_message("assistant-tail-should-drop"))
    agent.memory.add_message(
        Message.tool_message("tool-tail-should-drop", "bash", "tc1")
    )

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 0 recent" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "compact summary" in text
    assert "assistant-tail-should-drop" not in text
    assert "tool-tail-should-drop" not in text
    assert agent._run_llm_calls == 1
    assert agent._run_compact_llm_calls == 1
    assert agent.last_run_llm_calls == 1
    assert agent.last_run_compact_llm_calls == 1
    operations = list(
        (tmp_path / ".logs" / "agent_runtime_operations").glob("*.jsonl")
    )
    assert len(operations) == 1
    rows = [
        json.loads(line)
        for line in operations[0].read_text(encoding="utf-8").splitlines()
    ]
    intent = next(
        row for row in rows if row["kind"] == "llm" and row["phase"] == "intent"
    )
    assert intent["payload"]["turn_kind"] == "compact_inband"

@pytest.mark.asyncio
async def test_long_horizon_compact_preserves_protected_eda_prefix(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(
        Message.assistant_message("EDA command: inspect train.csv")
    )
    agent.memory.add_message(
        Message.tool_message("EDA output: train shape is (1944, 14)", "bash", "eda1")
    )
    agent.memory.add_message(
        Message.assistant_message("later exploration should compact away")
    )
    agent.memory.add_message(
        Message.tool_message("later output should compact away", "bash", "late1")
    )

    with caplog.at_level(logging.WARNING):
        info = agent._memory_ctx.set_protected_raw_prefix(
            3, warn_chars=1, label="test EDA prefix"
        )
    assert info["message_count"] == 2
    assert "exceeding warning threshold 1" in caplog.text

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 2 protected raw + 0 recent" in out
    assert agent._memory_ctx._protected_raw_prefix_end_index == 3
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "EDA command: inspect train.csv" in text
    assert "EDA output: train shape is (1944, 14)" in text
    assert "compact summary" in text
    assert "later exploration should compact away" not in text
    assert "later output should compact away" not in text

@pytest.mark.asyncio
async def test_long_horizon_compact_can_replace_protected_eda_with_facts_card(
    tmp_path: Path,
) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    large_scratch = "python3 -c " + repr("print('scratch eda')\n" * 200)
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "tc_eda",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": large_scratch}),
                    },
                }
            ],
        )
    )
    agent.memory.add_message(
        Message.tool_message(
            "train shape is (1944, 14)\nmissing values: 0\nFinal Validation Score: 0.77",
            "bash",
            "tc_eda",
        )
    )
    agent.memory.add_message(
        Message.assistant_message("later exploration should compact away")
    )

    info = agent._memory_ctx.replace_protected_raw_prefix_with_summary(
        3,
        "EDA facts:\n- train shape is (1944, 14)\n- missing values: 0",
        warn_chars=50_000,
        label="LHR protected EDA facts",
    )

    assert info["mode"] == "summary"
    assert info["message_count"] == 1
    assert info["original_chars"] > info["chars"]

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 1 protected raw + 0 recent" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "EDA facts:" in text
    assert "train shape is (1944, 14)" in text
    assert "scratch eda" not in text
    assert "Final Validation Score: 0.77" not in text
    assert "later exploration should compact away" not in text

def test_repl_environment_context_is_small_and_static_shape(tmp_path: Path) -> None:
    text = _repl_environment_context_prompt(tmp_path)
    assert text.startswith("<environment_context>")
    assert f"<cwd>{tmp_path.resolve()}</cwd>" in text
    assert "<shell>" in text
    assert "<current_date>" in text
    assert "validation" not in text.lower()
    assert "round" not in text.lower()

@pytest.mark.asyncio
async def test_science_agent_multiturn_memory(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="one"),
            SimpleNamespace(tool_calls=[], content="two"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    await agent.run("first")
    await agent.run("second")
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    texts = [
        r.memory_record.message.content
        for r in stored
        if r.memory_record.message.role == "user"
    ]
    joined = " ".join(str(t) for t in texts)
    assert "first" in joined and "second" in joined

def test_science_agent_mid_run_compact_enabled_flag(tmp_path: Path) -> None:
    """Config ``mid_run_compact_enabled`` is stored for run_loop mid-session compact."""
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        mid_run_compact_enabled=False,
    )
    assert agent._mid_run_compact_enabled is False

@pytest.mark.asyncio
async def test_run_mid_run_compact_triggers_when_window_omits_messages(
    tmp_path: Path,
) -> None:
    """When the REPL window overflows, run_loop compacts before the next LLM call."""
    calls: list[bool] = []

    async def fake_compact_inband(*, mid_run: bool = False) -> str:
        calls.append(mid_run)
        return "Compacted 3 messages into in-band summary (10 chars)."

    llm = _FakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="after compact"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=3,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    for i in range(16):
        agent.memory.add_message(
            Message.assistant_message(f"assistant-{i}: " + "x" * 1000),
        )
        agent.memory.add_message(
            Message.tool_message(f"tool-{i}: " + "y" * 1000, "bash", str(i)),
        )
    agent.compact_inband = fake_compact_inband  # type: ignore[method-assign]

    out = await agent.run("go")
    assert "after compact" in out
    assert calls == [True]

@pytest.mark.asyncio
async def test_run_context_limit_compact_retries_durable_when_inband_still_omits(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    omitted_values = [1, 1, 0]

    async def fake_compact_inband(*, mid_run: bool = False) -> str:
        calls.append(f"inband:{mid_run}")
        return "Compacted 3 messages into in-band summary (10 chars)."

    async def fake_compact(*, mid_run: bool = False) -> str:
        calls.append(f"durable:{mid_run}")
        return "Compacted 2 messages into summary (8 chars)."

    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="after fallback")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=1,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    agent._lnr_compact_on_context_threshold = True
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))

    def fake_build_messages_for_llm_with_stats() -> tuple[list[Message], int]:
        omitted = omitted_values.pop(0) if omitted_values else 0
        return [Message.user_message("visible context")], omitted

    agent._memory_ctx.build_messages_for_llm_with_stats = (
        fake_build_messages_for_llm_with_stats
    )
    agent.compact_inband = fake_compact_inband  # type: ignore[method-assign]
    agent.compact = fake_compact  # type: ignore[method-assign]

    out = await agent.run("go")

    assert "after fallback" in out
    assert calls == ["inband:True", "durable:True"]

@pytest.mark.asyncio
async def test_run_window_compact_preserves_first_user_prefix(tmp_path: Path) -> None:
    """Overflow compaction must not erase the original task / first user query."""

    llm = _RecordingFakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="summary of old work"),
            SimpleNamespace(tool_calls=[], content="after compact"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=1,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    first = "FIRST USER QUERY: solve nomad2018 and write submission.csv"
    agent.memory.add_message(Message.user_message(first))
    for _ in range(6):
        agent.memory.add_message(Message.assistant_message("x" * 5000))

    out = await agent.run("continue")
    assert "after compact" in out

    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in stored]
    assert first in contents[0]
    assert any("summary of old work" in c for c in contents)
    assert any(c == "continue" for c in contents)
    assert llm.kwarg_snapshots[0].get("tool_choice") == "none"
    assert "Internal context maintenance request" in (
        llm.message_snapshots[0][-1].content or ""
    )

@pytest.mark.asyncio
async def test_compact_clears_history(tmp_path: Path) -> None:
    async def summarize(_prompt: str, _system_prompt: str) -> str:
        return "summary text"

    mem = Memory(max_messages=50)
    mem.add_message(Message.user_message("old stuff"))
    from scienceflow.research.state.knowledge.context.memory_context import MemoryContextManager

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=10_000)
    out = await ctx.compact(completion=summarize)
    assert "Compacted" in out
    after = mem.chat_history_memory.retrieve(window_size=None)
    assert len(after) >= 1
    # New contract: leading user pin (msg[0]) is preserved verbatim across compact;
    # the LLM summary is appended *after* the pin instead of replacing it.
    contents = [r.memory_record.message.content or "" for r in after]
    assert "old stuff" in contents[0], "msg[0] head must survive compact verbatim"
    assert any("summary" in c.lower() for c in contents), (
        "summary must be appended after the pin"
    )

def test_replace_history_with_compacted_summary_drops_old_leading_summaries(
    tmp_path: Path,
) -> None:
    from scienceflow.research.state.knowledge.context.memory_context import (
        COMPACTED_CONVERSATION_SUMMARY_MARKER,
        MemoryContextManager,
    )

    mem = Memory(max_messages=50)
    mem.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    mem.add_message(
        Message.user_message(f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nold one"),
    )
    mem.add_message(
        Message.user_message(f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nold two"),
    )
    mem.add_message(Message.assistant_message("old assistant turn"))

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=10_000)
    out = ctx.replace_history_with_compacted_summary("new summary", recent_messages=0)

    assert "kept 1 leading + 0 recent" in out
    after = mem.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in after]
    assert contents == [
        "FIRST USER QUERY: keep me stable",
        f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nnew summary",
    ]
