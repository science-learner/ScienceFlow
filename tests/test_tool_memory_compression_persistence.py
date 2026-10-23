"""Tool Memory Compression contracts: persistence."""

from __future__ import annotations

from tests._tool_memory_compression_support import *  # noqa: F401,F403


def test_compress_write_tool_call_preserves_large_body_for_cache() -> None:
    body = "x" * 500
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": body}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["path"] == "a.py"
    assert args["content"] == body

def test_compress_write_tool_call_keeps_placeholder_attempt_for_storage() -> None:
    mimic = "[file content omitted - 51 chars written to disk]"
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": mimic}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["content"] == mimic

def test_compress_write_tool_call_keeps_new_angle_placeholder_attempt_for_storage() -> None:
    mimic = (
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>"
    )
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": mimic}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["content"] == mimic

def test_compress_write_short_body_unchanged() -> None:
    short = "x" * 100
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": short}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    args = json.loads(out.function.arguments)
    assert args["content"] == short

def test_compress_edit_tool_call_long_old_and_new_str() -> None:
    long_new = "y" * 300
    long_old = "o" * 240
    tc = SimpleNamespace(
        id="c2",
        type="function",
        function=SimpleNamespace(
            name="edit",
            arguments=json.dumps(
                {"path": "b.py", "old_str": long_old, "new_str": long_new},
            ),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    args = json.loads(out.function.arguments)
    assert args["old_str"].startswith("o" * 60)
    assert args["old_str"].endswith("o" * 60)
    assert (
        "# <<<MEMORY_COMPRESSED: edit.old_str=240bytes — NOT CODE — real content on disk>>>"
        in args["old_str"]
    )
    assert len(args["new_str"]) < len(long_new)
    assert (
        "# <<<MEMORY_COMPRESSED: omitted 180 chars from new_str — NOT CODE>>>" in args["new_str"]
    )
    assert args["new_str"].startswith("y" * 60)
    assert args["new_str"].endswith("y" * 60)

def test_compress_edit_success_output_strips_context() -> None:
    obs = (
        "Edited foo.py: 1 replacement OK.\n"
        "(context around edit near line 5)\n"
        ">>     1|line\n"
        "(42 lines, sha256~abcdef0123456789)"
    )
    c = compress_edit_success_output_for_memory(obs)
    assert "context around edit" not in c
    assert c == (
        "Edited foo.py: 1 replacement OK.\n"
        "(42 lines, sha256~abcdef0123456789)"
    )

def test_compress_edit_success_relaxed_regex_long_hash_and_trailing() -> None:
    """Last line may have longer hex and optional trailing metadata."""
    long_hash = "a" * 64
    obs = (
        "Edited foo.py: 1 replacement OK.\n"
        "(context)\n"
        f"(42 lines, sha256~{long_hash}) [verified]"
    )
    c = compress_edit_success_output_for_memory(obs)
    assert "context" not in c
    assert c == (
        "Edited foo.py: 1 replacement OK.\n"
        f"(42 lines, sha256~{long_hash}) [verified]"
    )

def test_compress_edit_success_returns_obs_on_internal_error() -> None:
    obs = "Edited x: 1 replacement OK.\n(1 lines, sha256~ab)"
    with mock.patch("scienceflow.research.state.knowledge.context.memory_context.re.match", side_effect=RuntimeError("boom")):
        assert compress_edit_success_output_for_memory(obs) == obs

def test_record_read_default_full_uses_snapshot_ref_after_write(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadref", "Draft", 200)
    ws = tmp_path / "wreadref"
    ws.mkdir()
    lines = ["def main():", "    return 1"]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (2 lines, 40 bytes, sha256~abc123)"),
    )
    raw = "[solution.py: 2 lines total, showing 1-2]\n     1|def main():\n     2|    return 1"

    obs = ctx.record_tool_result(
        "read",
        {"path": "solution.py"},
        ToolResult(output=raw),
        raw_id="tool_000002_read.txt",
    )

    assert "snapshot_ref_v1" in obs
    assert "[see current snapshot: solution.py sha~" in obs
    assert "def main():" not in obs
    assert "tool_000002_read.txt" in obs

def test_record_grep_uses_snapshot_ref_for_high_volume_matches(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mgrepref", "Draft", 200)
    ws = tmp_path / "wgrepref"
    ws.mkdir()
    lines = [f"target_{i} = {i}" for i in range(12)]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (12 lines, 200 bytes, sha256~abc123)"),
    )
    raw = "\n".join(f"solution.py:{i}:target_{i} = {i}" for i in range(1, 13))

    obs = ctx.record_tool_result(
        "grep",
        {"pattern": "target"},
        ToolResult(output=raw),
        raw_id="tool_000003_grep.txt",
    )

    assert "snapshot_ref_v1" in obs
    assert "solution.py: 12 matches (lines 1,2,3,4,5,6,7,8,9,10,...)" in obs
    assert "target_1 = 1" not in obs
    assert "tool_000003_grep.txt" in obs

def test_assistant_message_from_api_preserves_large_write_tool_call() -> None:
    body = "print('hello')\n" + ("x" * 600)
    syn = SimpleNamespace(
        content="",
        reasoning_content="write file",
        tool_calls=[
            SimpleNamespace(
                id="call_write",
                type="function",
                function=SimpleNamespace(
                    name="write",
                    arguments=json.dumps({"path": "solution.py", "content": body}),
                ),
            ),
        ],
    )
    msg = _assistant_message_from_api(syn)
    assert msg.reasoning_content == "write file"
    tc = msg.tool_calls[0]
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    args = json.loads(raw)
    assert args["path"] == "solution.py"
    assert args["content"] == body

def test_default_system_forbids_confirmatory_read_after_write_or_edit() -> None:
    assert "successful **write** or **edit**" in _DEFAULT_SYSTEM
    assert "double-check the write took effect" in _DEFAULT_SYSTEM

def test_default_system_explains_chat_memory_write_omit() -> None:
    assert "Historical write tool call omitted from executable LLM context" in _DEFAULT_SYSTEM
    assert "not a tool call and is not file content" in _DEFAULT_SYSTEM
    # Legacy formats still mentioned for backward-compat
    assert "WRITE_OK memory-compression" in _DEFAULT_SYSTEM
    assert "DO NOT COPY" in _DEFAULT_SYSTEM
    assert "chat-memory" in _DEFAULT_SYSTEM

def test_llm_view_keeps_large_write_tool_call_for_cache(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj", "Draft", 200)
    ws = tmp_path / "wproj"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    body = "print('hello')\n" + ("x" * 500)
    raw_args = {"path": "solution.py", "content": body, "thought": "write full file"}
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps(raw_args),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Written solution.py (12 lines, 520 bytes, sha256~abc123)",
            "write",
            "tc_write",
        ),
    )

    stored = mem.chat_history_memory.retrieve(window_size=None)
    stored_tc = stored[0].memory_record.message.tool_calls[0]
    stored_fn = stored_tc["function"] if isinstance(stored_tc, dict) else stored_tc.function
    stored_name = stored_fn["name"] if isinstance(stored_fn, dict) else stored_fn.name
    stored_raw = stored_fn["arguments"] if isinstance(stored_fn, dict) else stored_fn.arguments
    stored_args = json.loads(stored_raw)
    assert stored_name == "write"
    assert stored_args["content"] == body

    visible = ctx.build_messages_for_llm()
    assert visible[0].role == "assistant"
    assert getattr(visible[0], "tool_calls", None)
    visible_tc = visible[0].tool_calls[0]
    visible_fn = visible_tc["function"] if isinstance(visible_tc, dict) else visible_tc.function
    visible_raw = visible_fn["arguments"] if isinstance(visible_fn, dict) else visible_fn.arguments
    visible_args = json.loads(visible_raw)
    assert visible_args["path"] == "solution.py"
    assert visible_args["content"] == body
    assert "thought" not in visible_args
    assert visible[1].role == "tool"
    assert "Written solution.py" in (visible[1].content or "")

def test_llm_view_projects_placeholder_write_tool_call(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj_placeholder", "Draft", 200)
    ws = tmp_path / "wproj_placeholder"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mimic = "[WRITE_OK memory-compression: 13617 chars -> sha256:abc123; full body applied to disk; NOT file content]"
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": mimic}),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Write rejected: content looked like a placeholder",
            "write",
            "tc_write",
        ),
    )

    visible = ctx.build_messages_for_llm()
    assert visible[0].role == "assistant"
    assert not getattr(visible[0], "tool_calls", None)
    assert "unsafe historical write payloads" in (visible[0].content or "")
    assert visible[1].role == "user"
    assert "Historical write tool call omitted" in (visible[1].content or "")

def test_llm_view_keeps_large_write_and_snapshot_result(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj_map", "Draft", 200)
    ws = tmp_path / "wproj_map"
    ws.mkdir()
    sol = ws / "solution.py"
    body = "\n".join(
        [
            "import argparse",
            "import xgboost as xgb",
            "import pandas as pd",
            "",
            "def engineer_features(df):",
            "    return df",
            "",
            "def train_and_evaluate(args):",
            "    pd.DataFrame({'id': [1]}).to_csv('submission.csv', index=False)",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--epochs', type=int, default=1)",
            "    args = parser.parse_args()",
            "    train_and_evaluate(args)",
        ],
    )
    sol.write_text(body, encoding="utf-8")
    snap = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=1200,
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": body}),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Written solution.py (15 lines, 500 bytes, sha256~abc123)\n\n" + snap,
            "write",
            "tc_write",
        ),
    )

    visible = ctx.build_messages_for_llm()
    text = "\n".join(str(m.content or "") for m in visible)
    assert getattr(visible[0], "tool_calls", None)
    visible_tc = visible[0].tool_calls[0]
    visible_fn = visible_tc["function"] if isinstance(visible_tc, dict) else visible_tc.function
    visible_args = json.loads(
        visible_fn["arguments"] if isinstance(visible_fn, dict) else visible_fn.arguments
    )
    assert visible_args["content"] == body
    assert "[tool-summary code-map: solution.py]" in text
    assert "engineer_features(df):" in text
    assert "train_and_evaluate(args):" in text
    assert "--epochs" in text
    assert "submission.csv" in text

def test_write_auto_snapshot_uses_semantic_layers_for_python(tmp_path) -> None:
    sol = tmp_path / "solution.py"
    before = "\n".join(
        [
            "import pandas as pd",
            "",
            "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:",
            '    """Feature engineering pipeline."""',
            "    df = df.copy()",
            "    df['ratio'] = df['a'] / df['b'].clip(lower=1)",
            "    df['ratio_log'] = df['ratio']",
            "    return df",
            "",
            "def train_predict(train_df, test_df):",
            "    model = object()",
            "    features = engineer_features(train_df)",
            "    return features",
        ],
    )
    after = before.replace(
        "    df['ratio_log'] = df['ratio']",
        "    df['ratio_log'] = df['ratio'].clip(0, 100)",
    )
    sol.write_text(after, encoding="utf-8")

    block = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=8000,
        previous_text=before,
        tool_name="write",
        args={"path": "solution.py"},
    )

    assert len(block) <= 8000
    assert "[tool-summary code-map: solution.py]" in block
    assert "[changed-range excerpt: solution.py::engineer_features lines 3-8]" in block
    assert "df['ratio_log'] = df['ratio'].clip(0, 100)" in block
    assert "[symbol-level summary: solution.py]" in block
    assert "def train_predict(train_df,test_df):" in block
    assert "[source-excerpt: head]" not in block
    assert "middle lines omitted" not in block

def test_llm_view_projects_large_edit_payload_as_summary(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "meditproj", "Draft", 200)
    ws = tmp_path / "weditproj"
    ws.mkdir()
    sol = ws / "solution.py"
    body = "\n".join(
        [
            "def engineer_features(df):",
            "    df = df.copy()",
            "    df['x'] = df['x'].clip(0, 10)",
            "    return df",
        ],
    )
    sol.write_text(body, encoding="utf-8")
    snap = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=8000,
        tool_name="edit",
        args={"path": "solution.py", "new_str": "df['x'] = df['x'].clip(0, 10)"},
    )
    old_payload = "old-anchor\n" + ("x = 1\n" * 120)
    new_payload = "new-anchor\n" + ("x = 2\n" * 120)
    mem.add_message(
        Message(
            role="assistant",
            content="editing solution",
            tool_calls=[
                {
                    "id": "tc_edit",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps(
                            {
                                "path": "solution.py",
                                "old_str": old_payload,
                                "new_str": new_payload,
                            },
                        ),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Edited solution.py: 1 replacement OK.\n\n" + snap,
            "edit",
            "tc_edit",
        ),
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    visible = ctx.build_messages_for_llm()
    text = "\n".join(str(m.content or "") for m in visible)
    assert visible[0].role == "assistant"
    assert not getattr(visible[0], "tool_calls", None)
    assert "edit payloads" in (visible[0].content or "")
    assert "Historical edit tool call omitted" in text
    assert "Changed range: [changed-range excerpt: solution.py::engineer_features" in text
    assert "[tool-summary code-map: solution.py]" in text
    assert old_payload not in text
    assert new_payload not in text

def test_write_auto_snapshot_prefers_code_map_under_small_budget(tmp_path) -> None:
    sol = tmp_path / "solution.py"
    sol.write_text(
        "\n".join(
            [
                "import argparse",
                "import xgboost as xgb",
                "import pandas as pd",
                "",
                "def engineer_features(df):",
                "    return df",
                "",
                "def train_and_evaluate(args):",
                "    pd.DataFrame({'id': [1]}).to_csv('submission.csv', index=False)",
                "",
                "def main():",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--epochs', type=int, default=1)",
                "    args = parser.parse_args()",
                "    train_and_evaluate(args)",
            ],
        ),
        encoding="utf-8",
    )

    block = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=900,
    )
    assert len(block) <= 900
    assert "[tool-summary code-map: solution.py]" in block
    assert "engineer_features(df):" in block
    assert "train_and_evaluate(args):" in block
    assert "--epochs" in block
    assert "submission.csv" in block
    assert "Do NOT call" not in block

def test_looks_like_write_placeholder_mimicry() -> None:
    assert _looks_like_write_placeholder_mimicry("<4884 chars>")
    assert _looks_like_write_placeholder_mimicry("  <12 chars>  ")
    assert _looks_like_write_placeholder_mimicry("<0 chars>")
    assert _looks_like_write_placeholder_mimicry(
        "[file content omitted - 500 chars written to disk]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "  [file content omitted - 51 chars written to disk]  ",
    )
    assert _looks_like_write_placeholder_mimicry(
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>",
    )
    assert _looks_like_write_placeholder_mimicry(
        "[interaction-log: write body omitted; 13617 chars applied by tool]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [chat-memory: write payload omitted (500 bytes); "
        "following write tool result confirms on-disk file.]",
    )
    # Legacy "DO NOT COPY" format (still recognised for backward-compat)
    assert _looks_like_write_placeholder_mimicry(
        "[DO NOT COPY - display-only stub: real 13617-char body applied to disk]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [DO NOT COPY - display-only stub: real 500-byte solution body on disk; "
        "see following write tool result.]",
    )
    # New "WRITE_OK memory-compression" format (current)
    assert _looks_like_write_placeholder_mimicry(
        "[WRITE_OK memory-compression: 13617 chars -> sha256:abc1234567890abc; "
        "full body applied to disk; NOT file content]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [WRITE_OK memory-compression: 500 bytes -> sha256:def0987654321fed; "
        "the full file is on disk. This comment line is NOT file content; "
        "do NOT copy it as a new write payload. Use read tool only when you need the on-disk text "
        "to craft an edit or verify a crash.]",
    )
    mem_blob = (
        "x" * 60
        + "\n\n# [MEMORY_COMPRESSED: full file is 500 bytes; real content on disk]\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob)
    mem_blob_v2 = (
        "x" * 60
        + "\n\n# <<<MEMORY_COMPRESSED: prior successful write, 500 bytes already on disk — "
        "do NOT rewrite to 'fix' this; use read if you need the full current content>>>\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob_v2)
    mem_blob_write_ok = (
        "x" * 60
        + "\n\n# <<<WRITE_OK: 500 bytes written and syntax-validated — "
        "file is complete on disk. No read needed.>>>\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob_write_ok)
    assert _looks_like_write_placeholder_mimicry(_SCRUBBED_WRITE_PLACEHOLDER_NOTE)
    assert not _looks_like_write_placeholder_mimicry("print('hi')")
    assert not _looks_like_write_placeholder_mimicry("<not digits chars>")
    assert not _looks_like_write_placeholder_mimicry(
        "[file content omitted - not-a-number chars written to disk]",
    )
    long_with_marker = (
        "# real file\n" * 200 + "# [MEMORY_COMPRESSED: not a real marker line]\n" + "z\n" * 200
    )
    assert not _looks_like_write_placeholder_mimicry(long_with_marker)

def test_scrub_last_assistant_write_tool_call_content() -> None:
    bad = "[file content omitted - 51 chars written to disk]"
    rec = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": bad}),
                    },
                },
            ],
        },
    }
    memory = SimpleNamespace(
        chat_history_memory=SimpleNamespace(
            storage=SimpleNamespace(memory_list=[rec]),
        ),
    )
    scrub_last_assistant_write_tool_call_content(memory, tool_call_id="tc1")
    assert rec["message"]["tool_calls"][0]["function"]["name"] == "write"
    args = json.loads(rec["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == bad

def test_record_tool_write_appends_auto_snapshot_from_disk(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn", "Draft", 200)
    ws = tmp_path / "w_sn"
    ws.mkdir()
    (ws / "solution.py").write_text("print(1)\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = "Written solution.py (1 lines, 10 bytes, sha256~deadbeef)\n     1|not-on-disk"
    obs = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX in obs
    assert "print(1)" in obs
    assert "not-on-disk" not in obs

    ctx2 = MemoryContextManager(
        mem, ws, tool_memory_compression=True, write_auto_snapshot_enabled=False
    )
    obs2 = ctx2.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX not in obs2

def test_record_tool_write_auto_snapshots_any_python_file_with_signatures(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_py", "Draft", 200)
    ws = tmp_path / "w_sn_py"
    ws.mkdir()
    (ws / "utils.py").write_text(
        "def load_data(path, limit=100, *, strict=False):\n"
        "    return path\n\n"
        "class Trainer:\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n",
        encoding="utf-8",
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    obs = ctx.record_tool_result(
        "write",
        {"path": "utils.py"},
        ToolResult(output="Written utils.py (6 lines, 120 bytes, sha256~deadbeef)"),
    )

    assert _AUTO_SNAPSHOT_PREFIX in obs
    assert "[tool-summary code-map: utils.py]" in obs
    assert "load_data(path,limit=100,*,strict=False)" in obs
    assert "class Trainer" in obs
    assert "fit(self,X,y=None)" in obs

def test_record_tool_write_does_not_snapshot_non_code_outputs(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_csv", "Draft", 200)
    ws = tmp_path / "w_sn_csv"
    ws.mkdir()
    (ws / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    obs = ctx.record_tool_result(
        "write",
        {"path": "submission.csv"},
        ToolResult(output="Written submission.csv (2 lines, 16 bytes, sha256~deadbeef)"),
    )

    assert _AUTO_SNAPSHOT_PREFIX not in obs

def test_record_tool_write_dedupes_unchanged_auto_snapshot(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_dedup", "Draft", 200)
    ws = tmp_path / "w_sn_dedup"
    ws.mkdir()
    (ws / "solution.py").write_text("print(1)\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = "Written solution.py (1 lines, 10 bytes, sha256~deadbeef)\n     1|not-on-disk"

    obs1 = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    obs2 = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))

    assert obs1.count(_AUTO_SNAPSHOT_PREFIX) == 1
    assert _AUTO_SNAPSHOT_PREFIX not in obs2

def test_record_tool_edit_strips_raw_snapshot_and_appends_canonical_once(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "medsn", "Draft", 200)
    ws = tmp_path / "e_sn"
    ws.mkdir()
    (ws / "solution.py").write_text("a = 1\nb = 99\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = (
        "Edited solution.py: 1 replacement OK.\n"
        "context around edit\n"
        "(2 lines, sha256~deadbeef)\n"
        "--- Current file snapshot (after edit) ---\n"
        "     1|raw tool snapshot should be dropped\n"
    )

    obs = ctx.record_tool_result("edit", {"path": "solution.py"}, ToolResult(output=raw))

    assert "context around edit" not in obs
    assert "--- Current file snapshot (after edit) ---" not in obs
    assert "raw tool snapshot should be dropped" not in obs
    assert obs.count(_AUTO_SNAPSHOT_PREFIX) == 1
    assert "     2|b = 99" in obs

    obs2 = ctx.record_tool_result("edit", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX not in obs2
    assert "--- Current file snapshot (after edit) ---" not in obs2

def test_record_tool_write_clears_read_overlap_state(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mclr", "Draft", 200)
    ws = tmp_path / "wclr"
    ws.mkdir()
    p = ws / "x.txt"
    lines = [f"row{i}" for i in range(10)]
    p.write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=20)

    def _fmt(nl: int, off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        if lim <= 0:
            end = nl
        else:
            end = min(nl, s0 + lim)
        chunk = lines[:nl][s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[x.txt: {nl} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING not in o1
    # rewrite same file path — clears read coverage for x.txt
    lines = [f"new{i}" for i in range(10)]
    p.write_text("\n".join(lines), encoding="utf-8")
    wout = "Written x.txt (10 lines, 100 bytes, sha256~a)\n" + "\n".join(
        f"  {i:4d}|{lines[i-1]}" for i in range(1, 11)
    )
    ctx.record_tool_result("write", {"path": "x.txt"}, ToolResult(output=wout))
    o2 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING not in o2
    o3 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING in o3

def test_scrub_last_assistant_write_tool_call_content_new_placeholder() -> None:
    bad = (
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>"
    )
    rec = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": bad}),
                    },
                },
            ],
        },
    }
    memory = SimpleNamespace(
        chat_history_memory=SimpleNamespace(
            storage=SimpleNamespace(memory_list=[rec]),
        ),
    )
    scrub_last_assistant_write_tool_call_content(memory, tool_call_id="tc1")
    assert rec["message"]["tool_calls"][0]["function"]["name"] == "write"
    args = json.loads(rec["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == bad
