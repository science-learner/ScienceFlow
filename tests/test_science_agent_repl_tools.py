"""Science Agent Repl contracts: tools."""

from __future__ import annotations

from tests._science_agent_repl_support import *  # noqa: F401,F403


def test_bash_write_core_guides_large_observation_summaries() -> None:
    text = _code_agent_core_prompt(bash_file_write_mode=True)
    assert "do not use raw `cat`, recursive `ls`, or broad `find` output" in text
    assert "summary that scans the full target" in text
    assert "shape, columns, dtypes, missing values" in text
    assert "extension/type counts" in text
    assert "summarized or truncated bash result is not enough" in text
    assert "write the full log to a workspace file" in text
    assert "Track the current best observed metric" in text
    assert "Do not run verbose training or validation as `command | tail`" in text
    assert "After every substantive training/validation command returns" in text
    assert "immediately update a workspace ledger" in text
    assert "promote the best known artifact" in text
    assert "run the matching prediction/export command" in text
    assert "Nomad2018" not in text
    assert "submission.csv" not in text

def test_bash_write_core_and_prefix_are_generic() -> None:
    core = _code_agent_core_prompt(bash_file_write_mode=True)
    prefix = _repl_bash_file_write_prompt()
    assert "modify files through bash" in core
    assert "exact rewrite operation" in core
    assert (
        "Available tools in this REPL session are `bash`, `read`, `grep`, `glob`, and `ls`"
        in prefix
    )
    assert "Do not call tools named" not in prefix
    assert "`edit`" not in prefix
    assert "exact rewrite operation" in prefix
    assert "A bash observation may be summarized, deduplicated, or truncated" in prefix
    assert "recover exact facts" in prefix
    assert "record the current best metric" in prefix
    assert "command | tail" in prefix
    assert "workspace ledger" in prefix
    assert "update that ledger before" in prefix
    assert "preserve the matching artifact" in prefix
    assert "align any metric file" in prefix
    assert "cat >" not in core
    assert "cat >>" not in prefix
    assert "python3 - <<" not in core
    assert "python3 - <<" not in prefix
    assert "Nomad2018" not in core
    assert "Nomad2018" not in prefix
    assert "solution.py" not in core
    assert "solution.py" not in prefix
    assert "submission.csv" not in core
    assert "submission.csv" not in prefix

def test_repl_bash_write_mode_removes_write_edit_tools(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
        include_write_edit_tools=False,
    )
    names = set(agent.availableTools.tool_map)
    assert {"bash", "read", "grep", "glob", "ls"}.issubset(names)
    assert "write" not in names
    assert "edit" not in names

    runtime = agent._build_system_prompt()
    assert (
        "Available tools in this REPL mode are `bash`, `read`, `grep`, `glob`, and `ls`"
        in runtime
    )
    assert "Tool names must come from that available-tools list" in runtime
    assert "Do not call tools named" not in runtime
    assert "`edit`" not in runtime
    assert "exact rewrite operation" in runtime
    assert "cat >" not in runtime
    assert "**read** before **edit**" not in runtime

    messages = agent._build_system_messages()
    assert len(messages) == 2
    assert "modify files through bash" in (messages[0].content or "")

@pytest.mark.asyncio
async def test_repl_bash_write_mode_normalizes_file_tool_to_bash(
    tmp_path: Path,
) -> None:
    (tmp_path / "solution.py").write_text("print('old')\n")
    llm = _FakeLLM(
        [
            _tool_msg(
                "edit",
                {
                    "path": "solution.py",
                    "old_str": "print('old')",
                    "new_str": "print('new')",
                },
            ),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        include_write_edit_tools=False,
    )

    out = await agent.run("modify a file")

    assert "done" in out
    assert (tmp_path / "solution.py").read_text() == "print('new')\n"
    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    assert not any(
        m.role == "user" and "Unavailable tool call blocked" in str(m.content)
        for m in stored
    )
    assistant_tool_names = [
        (tc["function"]["name"] if isinstance(tc, dict) else tc.function.name)
        for m in stored
        if m.role == "assistant"
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    assert "edit" not in assistant_tool_names
    assert "bash" in assistant_tool_names
    tool_names = [getattr(m, "name", "") for m in stored if m.role == "tool"]
    assert "edit" not in tool_names
    assert "bash" in tool_names

def test_repl_bash_write_mode_bash_schema_mentions_file_changes(tmp_path: Path) -> None:
    tools = create_tool_collection(tmp_path, include_write_edit_tools=False).to_params()
    bash_tool = next(t for t in tools if t["function"]["name"] == "bash")
    desc = bash_tool["function"]["description"]
    command_desc = bash_tool["function"]["parameters"]["properties"]["command"][
        "description"
    ]
    assert "file creation or file modification" in desc
    assert "complete file content" in desc
    assert "exact rewrite operation" in desc
    assert "do not dump raw content" in desc
    assert "read`, `grep`, `glob`, and `ls`" in desc
    assert "A compact or truncated bash result is not complete ground truth" in desc
    assert "write verbose logs to workspace files" in desc
    assert "save the full log first" in desc
    assert "update a workspace ledger immediately" in desc
    assert "preserve improved best-known artifacts" in desc
    assert "promote the best known artifact" in desc
    assert "full-scan summaries" in command_desc
    assert "recover exact details" in command_desc
    assert "do not use `command | tail`" in command_desc
    assert "For file changes" in command_desc
    assert "edit" not in desc.lower()

def test_repl_bash_output_caps_are_configurable_on_tool_collection(
    tmp_path: Path,
) -> None:
    tools = create_tool_collection(
        tmp_path,
        include_write_edit_tools=False,
        max_bash_output_chars=1234,
        max_bash_stream_line_chars=321,
        bash_observation_summary_enabled=True,
    )
    bash_tool = tools.tool_map["bash"]
    assert bash_tool.max_output_chars == 1234
    assert bash_tool.max_stream_line_chars == 321
    assert bash_tool.observation_summary_enabled is True

@pytest.mark.asyncio
async def test_repl_bash_observation_summary_preserves_raw_artifact(
    tmp_path: Path,
) -> None:
    lines = [f"row-{i:03d},value-{i:03d}" for i in range(300)]
    (tmp_path / "large.csv").write_text("\n".join(lines) + "\n")
    llm = _FakeLLM(
        [
            _tool_msg("bash", {"command": "cat large.csv"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        bash_max_output_chars=500,
        bash_observation_summary_enabled=True,
    )

    await agent.run("inspect large file")

    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    tool_texts = [
        m.content or ""
        for m in stored
        if m.role == "tool" and getattr(m, "name", "") == "bash"
    ]
    assert tool_texts
    assert "bash_observation_head_tail_v1" in tool_texts[-1]
    assert "raw_id=tool_000001_bash.txt" in tool_texts[-1]
    assert "path=tool_outputs/tool_000001_bash.txt" not in tool_texts[-1]
    assert "row-000,value-000" in tool_texts[-1]
    assert "row-299,value-299" in tool_texts[-1]
    assert "row-150,value-150" not in tool_texts[-1]

    raw = find_node_log_path(tmp_path, "tool_outputs") / "tool_000001_bash.txt"
    raw_text = raw.read_text()
    assert "row-000,value-000" in raw_text
    assert "row-150,value-150" in raw_text
    assert "row-299,value-299" in raw_text
    assert not (tmp_path / "tool_outputs" / "tool_000001_bash.txt").exists()

def test_repl_bash_write_mode_rewrites_write_tool_nudges() -> None:
    text = (
        "Plan directly from `description.md` + `dataset/`, "
        "then **`write` `solution.py`**.\n"
        "You must **`write` `solution.py`** now (minimal pipeline is OK)."
    )
    out = _adapt_write_tool_hint_for_bash_only(text)
    assert "`write` `solution.py`" not in out
    assert "single bash command" in out
    assert "heredoc" not in out

def test_bash_command_classifier_covers_file_actions() -> None:
    assert (
        classify_bash_command("cat > solution.py <<'PYEOF'\nprint(1)\nPYEOF") == "write"
    )
    assert (
        classify_bash_command("cat >> ml_run_results.md <<'EOF'\nok\nEOF") == "append"
    )
    assert (
        classify_bash_command(
            "python3 - <<'PY'\nfrom pathlib import Path\n"
            "p=Path('solution.py')\ns=p.read_text()\n"
            "p.write_text(s.replace('a','b'))\nPY",
        )
        == "edit"
    )
    assert classify_bash_command("cp solution.py solution_v2.py") == "copy"
    assert classify_bash_command("python3 solution.py") == "run_solution"

def test_interaction_log_bash_tool_call_includes_kind() -> None:
    lines = format_tool_call_lines_for_interaction_log(
        "bash",
        {"command": "cat > solution.py <<'PYEOF'\nprint(1)\nPYEOF"},
        full=False,
    )
    assert len(lines) == 1
    assert '"bash_kind": "write"' in lines[0]

@pytest.mark.asyncio
async def test_science_agent_bash_then_text(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            _tool_msg("bash", {"command": "echo roundtrip"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    out = await agent.run("run echo")
    assert "done" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    roles = [r.memory_record.message.role for r in stored]
    assert "user" in roles
    assert "tool" in roles

@pytest.mark.asyncio
async def test_repl_write_result_omits_auto_snapshot_but_keeps_tool_call_code(
    tmp_path: Path,
) -> None:
    llm = _RecordingFakeLLM(
        [
            _tool_msg("write", {"path": "solution.py", "content": "print('ok')\n"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        write_auto_snapshot_enabled=False,
    )

    await agent.run("write solution")

    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    write_tool_results = [
        m.content or ""
        for m in stored
        if m.role == "tool" and getattr(m, "name", "") == "write"
    ]
    assert write_tool_results
    assert "File `solution.py` written successfully" in write_tool_results[-1]
    assert "[auto-snapshot after successful write:" not in write_tool_results[-1]

    assistant_tool_calls = [
        tc
        for m in stored
        for tc in (getattr(m, "tool_calls", None) or [])
        if m.role == "assistant"
    ]
    assert assistant_tool_calls
    tc = assistant_tool_calls[-1]
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    assert "print('ok')" in args

@pytest.mark.asyncio
async def test_science_agent_saves_tool_output_txt_artifact(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            _tool_msg(
                "bash", {"command": "python3 -c \"print('raw artifact check')\""}
            ),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )

    await agent.run("run echo")

    out_dir = find_node_log_path(tmp_path, "tool_outputs")
    raw = out_dir / "tool_000001_bash.txt"
    assert raw.is_file()
    assert "raw artifact check" in raw.read_text()
    idx = (out_dir / "index.txt").read_text()
    assert "raw_id=tool_000001_bash.txt" in idx
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    tool_messages = [
        r.memory_record.message.content
        for r in stored
        if r.memory_record.message.role == "tool"
    ]
    assert any("raw_id=tool_000001_bash.txt" in str(m) for m in tool_messages)

def test_workspace_source_git_auto_checkpoint_can_write_submission_snapshots_outside_workspace(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    workspace = tmp_path / "workspace"
    logs = tmp_path / ".logs"
    workspace.mkdir()
    (workspace / "train.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(workspace, track_globs=["*.py", "*.md"])

    (workspace / "submission.csv").write_text("id,y\n1,0.7\n", encoding="utf-8")
    result = auto_checkpoint_workspace_source(
        workspace,
        track_globs=["*.py", "*.md"],
        tool_name="lhr_stage_s01",
        metric_value_override=0.0612,
        stage_id="S01",
        submission_snapshot_dir=logs / "submission_snapshots",
    )

    assert result.ready is True
    assert result.submission_snapshot.startswith("../.logs/submission_snapshots/")
    assert not (workspace / "submission_snapshots").exists()
    assert (workspace / result.submission_snapshot).read_text(
        encoding="utf-8"
    ) == "id,y\n1,0.7\n"
    ledger = workspace / ".scienceflow_checkpoints" / "ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert row["submission_snapshot"] == result.submission_snapshot
