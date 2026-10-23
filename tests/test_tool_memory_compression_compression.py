"""Tool Memory Compression contracts: compression."""

from __future__ import annotations

from tests._tool_memory_compression_support import *  # noqa: F401,F403


def test_compress_tool_call_disabled_returns_original() -> None:
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": "x" * 100}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=False)
    assert out is tc

def test_compress_bash_dedup_repeated_lines_before_tail() -> None:
    spam = "[LightGBM] [Warning] noisy"
    body = "\n".join([spam] * 50 + [f"L{i:03d}" for i in range(30)])
    obs = "[exit=0, 1.0s]\n" + body
    c = compress_bash_tool_output_for_memory(obs, tail_lines=40, dedup_enabled=True)
    assert spam in c
    assert c.count(spam) == 1
    assert "log-dedup" in c
    assert "L029" in c
    assert len(c) < len(obs)

def test_compress_bash_success_keeps_tail_only() -> None:
    body = "\n".join(f"L{i:03d}" for i in range(30))
    obs = "[exit=0, 1.0s]\n" + body
    c = compress_bash_tool_output_for_memory(obs, tail_lines=5)
    assert "30 lines total" in c
    assert "L029" in c
    assert "L000" not in c
    # Header must reconstruct outer brackets only (not str.strip("[]") charset stripping).
    assert c.splitlines()[0].startswith("[exit=0, 1.0s]")

def test_compress_bash_short_output_unchanged() -> None:
    obs = "[exit=0, 0.1s]\nhello\nworld"
    assert compress_bash_tool_output_for_memory(obs, tail_lines=20) == obs

def test_record_bare_solution_bash_success_uses_short_tail(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mbare", "Draft", 200)
    ws = tmp_path / "wbare"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, bash_success_tail_lines=20)
    body = "\n".join([f"L{i:02d}" for i in range(20)] + ["Final Validation Score: 0.123"])
    obs = ctx.record_tool_result(
        "bash",
        {"command": "python3 solution.py"},
        ToolResult(output="[exit=0, 4.8s]\n" + body),
    )
    assert "bash_training_signal_v2" in obs
    assert "L00" not in obs
    assert "L19" in obs
    assert "Final Validation Score: 0.123" in obs

def test_record_bare_solution_bash_failure_preserves_traceback(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mtrace", "Draft", 200)
    ws = tmp_path / "wtrace"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, bash_success_tail_lines=20)
    body = "\n".join(
        [f"[DATA] Fold {i % 5} metric={0.08 + i / 10000:.5f}" for i in range(24)]
        + [
            "[DATASET] Generating submission...",
            "Traceback (most recent call last):",
            '  File "solution.py", line 256, in <module>',
            '    sub_df[target] = test_preds[target].values.astype("float64")',
            "AttributeError: 'numpy.ndarray' object has no attribute 'values'",
        ]
    )
    obs = ctx.record_tool_result(
        "bash",
        {"command": "python3 solution.py"},
        ToolResult(output="[exit=1, 3.0s]\n" + body),
    )
    assert "bash_training_signal_v2" in obs
    assert "preserved error/traceback lines" in obs
    assert "AttributeError" in obs
    assert "test_preds[target].values" in obs

def test_record_bash_dynamic_tail_solution_positive_and_substring_taskset(tmp_path) -> None:
    obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="python3 solution.py",
        line_count=30,
    )
    assert "bash_training_signal_v2" in obs
    assert "L00" not in obs
    assert "L29" in obs

    taskset_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="taskset -c 0 python3 solution.py",
        line_count=30,
    )
    assert "bash_training_signal_v2" in taskset_obs
    assert "L00" not in taskset_obs
    assert "L29" in taskset_obs

def test_record_bash_dynamic_tail_sampled_solution_uses_fallback(tmp_path) -> None:
    obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="NROWS=100 python3 solution.py",
    )
    assert "showing last 6" in obs
    assert "L13" not in obs
    assert "L14" in obs

def test_record_bash_dynamic_tail_test_readonly_install_and_fallback(tmp_path) -> None:
    pytest_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="CUDA_VISIBLE_DEVICES=0 pytest tests/test_x.py",
        line_count=130,
    )
    assert "bash_pytest_summary_v2" in pytest_obs

    py3_pytest_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="python3 -m pytest tests/test_x.py",
        line_count=130,
    )
    assert "bash_pytest_summary_v2" in py3_pytest_obs

    readonly_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="cat notes.txt",
    )
    assert "showing last 4" in readonly_obs

    install_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="pip install x",
    )
    assert "bash_install_summary_v2" in install_obs

    uv_run_install_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="uv run pip install x",
    )
    assert "showing last 6" in uv_run_install_obs

def test_message_char_len_includes_tool_calls_json() -> None:
    m = Message(
        role="assistant",
        content="hi",
        tool_calls=[
            {
                "id": "x",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"path":"p","content":"[file content omitted - 3 chars written to disk]"}',
                },
            },
        ],
    )
    assert _message_char_len(m) > len("hi")

def test_tool_call_invalid_json_unchanged() -> None:
    tc = SimpleNamespace(
        id="c",
        type="function",
        function=SimpleNamespace(name="write", arguments="not json"),
    )
    assert _compress_tool_call_for_memory(tc, enabled=True) is tc

def test_compress_bash_returns_obs_on_internal_error() -> None:
    """If compression fails, return original *obs* unchanged (never raise)."""

    class BoomSplitlines(str):
        def splitlines(self, *args, **kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("boom")

    body = "\n".join(f"L{i}" for i in range(25))
    obs = BoomSplitlines(f"[exit=0, 1.0s]\n{body}")
    out = compress_bash_tool_output_for_memory(obs, tail_lines=5)
    assert out is obs

def test_compress_tool_call_swallows_exception_returns_tc() -> None:
    class Boom:
        def __getattribute__(self, name: str):
            raise RuntimeError("boom")

    tc = SimpleNamespace(function=Boom())
    assert _compress_tool_call_for_memory(tc, enabled=True) is tc

def test_compress_read_short_file_unchanged() -> None:
    header = "[solution.py: 20 lines total]"
    body = "\n".join(f"    {i}|line {i}" for i in range(1, 21))
    obs = header + "\n" + body
    assert compress_read_output_for_memory(obs, max_lines=80) == obs

def test_compress_read_long_file_truncated() -> None:
    header = "[solution.py: 200 lines total, showing 1-200]"
    body_lines = [f"    {i}|line {i}" for i in range(1, 201)]
    obs = header + "\n" + "\n".join(body_lines)
    out = compress_read_output_for_memory(obs, max_lines=30)
    out_lines = out.splitlines()
    # Header preserved
    assert out_lines[0] == header
    # Only 30 content lines kept + truncation notice
    assert len(out_lines) == 32  # header + 30 body + truncation
    assert "truncated for LLM context" in out_lines[-1]
    assert "first 30 of 200" in out_lines[-1]

def test_compress_read_no_header_unchanged() -> None:
    """If first line doesn't start with '[', return unchanged."""
    obs = "some weird output\nline2"
    assert compress_read_output_for_memory(obs, max_lines=1) == obs

def test_compress_read_returns_obs_on_internal_error() -> None:
    class BoomSplitlines(str):
        def splitlines(self, *args, **kwargs):
            raise RuntimeError("boom")

    obs = BoomSplitlines("[file.py: 100 lines total]\nstuff")
    out = compress_read_output_for_memory(obs, max_lines=5)
    assert out is obs

def test_record_read_default_full_python_uses_code_map(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadmap", "Draft", 200)
    ws = tmp_path / "wreadmap"
    ws.mkdir()
    lines = [
        "import pandas as pd",
        "",
        "def engineer_features(df):",
        "    df = df.copy()",
        "    return df",
        "",
        "def train_predict(train, test):",
        "    return engineer_features(test)",
    ]
    lines += [f"# filler {i}" for i in range(80)]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    read_body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(lines[:60], start=1))
    raw = f"[solution.py: {len(lines)} lines total, showing 1-60]\n{read_body}"
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=20,
    )

    obs = ctx.record_tool_result(
        "read",
        {"path": "solution.py"},
        ToolResult(output=raw),
        raw_id="tool_000001_read.txt",
    )

    assert "read_code_map_v2" in obs
    assert "[tool-summary code-map: solution.py]" in obs
    assert "engineer_features(df):" in obs
    assert "# filler 70" not in obs
    assert "tool_000001_read.txt" in obs

def test_compress_bash_tool_call_short_command_unchanged() -> None:
    tc = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"command": "ls -la"}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out is tc

def test_compress_bash_tool_call_long_command_preserved_for_cache() -> None:
    long_cmd = "echo " + "x" * 600
    tc = SimpleNamespace(
        id="c2",
        type="function",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"command": long_cmd}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out is tc

def test_default_system_has_reasoning_discipline_and_placeholder_ban() -> None:
    assert "## Reasoning discipline" in _DEFAULT_SYSTEM
    assert "placeholder" in _DEFAULT_SYSTEM.lower()
    assert "MEMORY_COMPRESSED" in _DEFAULT_SYSTEM
    assert "thought" in _DEFAULT_SYSTEM.lower()
    assert "required" in _DEFAULT_SYSTEM.lower() or "`thought`" in _DEFAULT_SYSTEM
    assert "1–2 concise sentences" in _DEFAULT_SYSTEM
    assert "2–3 sentences" not in _DEFAULT_SYSTEM
    assert "Two to three" not in _DEFAULT_SYSTEM

def test_llm_view_regular_tool_projection_keeps_tool_call_serializable(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj2", "Draft", 200)
    ws = tmp_path / "wproj2"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mem.add_message(
        Message(
            role="assistant",
            content="checking files",
            tool_calls=[
                {
                    "id": "tc_bash",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "ls dataset", "thought": "internal note"},
                        ),
                    },
                },
            ],
        ),
    )
    mem.add_message(Message.tool_message("dataset files listed", "bash", "tc_bash"))

    visible = ctx.build_messages_for_llm()
    assert visible[0].tool_calls
    assert hasattr(visible[0].tool_calls[0], "model_dump")
    serialized = visible[0].to_dict()
    args = json.loads(serialized["tool_calls"][0]["function"]["arguments"])
    assert args == {"command": "ls dataset"}

def test_bash_is_pure_read_only_whitelist() -> None:
    assert _bash_is_pure_read_only("cat a.py") is True
    assert _bash_is_pure_read_only("head -n 50 a.py | grep foo") is True
    assert _bash_is_pure_read_only("ENV=1 head a.py") is True
    assert _bash_is_pure_read_only("cat a.py | sed -n '1,5p'") is True
    assert _bash_is_pure_read_only("sed -i 's/a/b/' x.py") is False
    assert _bash_is_pure_read_only("cat > solution.py << 'PY'\nprint(1)\nPY") is False
    assert _bash_is_pure_read_only("grep foo solution.py > matches.txt") is False
    assert _bash_is_pure_read_only("cat a.py && ls") is False
    assert _bash_is_pure_read_only("python3 a.py") is False

def test_bash_command_dumps_python_source_heuristic() -> None:
    assert bash_command_dumps_python_source("cat -n solution.py") is True
    assert bash_command_dumps_python_source("cat -n solution.py | head -174") is True
    assert bash_command_dumps_python_source("cat > solution.py << 'PY'\nprint(1)\nPY") is False
    assert bash_command_dumps_python_source("head -5 dataset/train.csv") is False
    assert bash_command_dumps_python_source("grep foo solution.py") is False

def test_record_tool_result_bash_cat_solution_appends_read_coaching(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadco", "Draft", 200)
    ws = tmp_path / "wreadco"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=50,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i:03d}" for i in range(80))
    o1 = ctx.record_tool_result("bash", {"command": "cat -n solution.py"}, ToolResult(output=long_out))
    assert "showing last" in o1
    assert "`read` tool" in o1

def test_record_tool_result_repeat_bash_cat_second_call_uncompressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m", "Draft", 200)
    ws = tmp_path / "w"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=50,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i:03d}" for i in range(80))
    tr = ToolResult(output=long_out)
    cmd = {"command": "cat solution.py"}
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "bash:" in o1 and "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "c1"))
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" not in o2
    assert "L000" in o2 and "L079" in o2

def test_record_tool_result_repeat_read_second_call_uncompressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m2", "Draft", 200)
    ws = tmp_path / "w2"
    ws.mkdir()
    p = ws / "f.txt"
    body = "\n".join(f"line-{i}" for i in range(300))
    p.write_text(
        "[f.txt: 300 lines total, showing 1-300]\n" + body,
        encoding="utf-8",
    )
    raw_read = p.read_text(encoding="utf-8")
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=40,
    )
    args = {"path": "f.txt"}
    o1 = ctx.record_tool_result("read", args, ToolResult(output=raw_read))
    assert "read output truncated" in o1
    mem.add_message(Message.tool_message(o1, "read", "r1"))
    o2 = ctx.record_tool_result("read", args, ToolResult(output=raw_read))
    assert "read output truncated" not in o2
    assert "line-0" in o2 and "line-299" in o2

def test_record_tool_result_read_different_offset_not_treated_as_repeat(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m3", "Draft", 200)
    ws = tmp_path / "w3"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=20,
    )
    n = 80
    a = f"[a: {n} lines total, showing 1-{n}]\n" + "\n".join(f"row-{i}" for i in range(n))
    o1 = ctx.record_tool_result("read", {"path": "a", "offset": 1}, ToolResult(output=a))
    assert "read output truncated" not in o1
    assert "row-79" in o1
    mem.add_message(Message.tool_message(o1, "read", "x1"))
    o2 = ctx.record_tool_result("read", {"path": "a", "offset": 2}, ToolResult(output=a))
    assert "read output truncated" not in o2

def test_record_tool_result_non_readonly_bash_repeat_still_compressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m4", "Draft", 200)
    ws = tmp_path / "w4"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=4,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(40))
    tr = ToolResult(output=long_out)
    cmd = {"command": "ls -la /tmp"}
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "b1"))
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o2

def test_gc_supersedes_first_compressed_after_different_tool(tmp_path) -> None:
    from inquirycraft.memory import Function, ToolCall

    mem = create_agent_memory(tmp_path / "mgc", "Draft", 200)
    ws = tmp_path / "wgc"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=3,
        bash_success_tail_lines_readonly=3,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(25))
    tr = ToolResult(output=long_out)
    cmd = {"command": "cat solution.py"}

    def _asst_bash(tcid: str) -> None:
        mem.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=tcid,
                        function=Function(
                            name="bash",
                            arguments=json.dumps(cmd),
                        ),
                    ),
                ],
            ),
        )

    _asst_bash("t1")
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "t1"))

    _asst_bash("t2")
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" not in o2
    mem.add_message(Message.tool_message(o2, "bash", "t2"))

    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="te",
                    function=Function(
                        name="edit",
                        arguments=json.dumps({"path": "x.py", "old_str": "a", "new_str": "b"}),
                    ),
                ),
            ],
        ),
    )
    ctx.record_tool_result(
        "edit",
        {"path": "x.py", "old_str": "a", "new_str": "b"},
        ToolResult(output="Edited x.py: 1 replacement OK.\ncontext\n(1 lines, sha256~abc)"),
    )

    records = mem.chat_history_memory.storage.load()
    first_tool = next(
        r
        for r in records
        if isinstance(r, dict)
        and isinstance(r.get("message"), dict)
        and r["message"].get("role") == "tool"
        and r["message"].get("tool_call_id") == "t1"
    )
    content = str(first_tool["message"].get("content") or "")
    assert "superseded" in content.lower()

def test_record_tool_read_overlap_nudge_subrange(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mrov", "Draft", 200)
    ws = tmp_path / "wrov"
    ws.mkdir()
    p = ws / "x.txt"
    all_lines = [f"line{i}" for i in range(80)]
    p.write_text("\n".join(all_lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=200)

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 80
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = all_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[x.txt: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read",
        {"path": "x.txt", "offset": 1, "limit": 40},
        ToolResult(output=_fmt(1, 40)),
    )
    assert _READ_OVERLAP_COACHING not in o1
    o2 = ctx.record_tool_result(
        "read",
        {"path": "x.txt", "offset": 20, "limit": 5},
        ToolResult(output=_fmt(20, 5)),
    )
    assert _READ_OVERLAP_COACHING in o2

def test_record_tool_redundant_read_solution_py_uses_coverage_summary(tmp_path) -> None:
    """Tier 2: redundant read of solution.py records coverage, not repeated source."""
    mem = create_agent_memory(tmp_path / "msil", "Draft", 200)
    ws = tmp_path / "wsil"
    ws.mkdir()
    sol = ws / "solution.py"
    sol_lines = [f"# line {i}" for i in range(40)]
    sol.write_text("\n".join(sol_lines), encoding="utf-8")
    ctx = MemoryContextManager(
        mem, ws,
        tool_memory_compression=True,
        read_success_max_lines=200,
        write_auto_snapshot_enabled=True,
        write_auto_snapshot_paths=("solution.py",),
        write_auto_snapshot_max_lines=400,
        write_auto_snapshot_max_chars=30_000,
    )

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 40
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = sol_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[solution.py: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 1, "limit": 40},
        ToolResult(output=_fmt(1, 40)),
    )
    assert _AUTO_SNAPSHOT_PREFIX not in o1, "first read must not get a snapshot block"
    assert _READ_OVERLAP_COACHING not in o1
    o2 = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 5, "limit": 10},
        ToolResult(output=_fmt(5, 10)),
    )
    assert _AUTO_SNAPSHOT_PREFIX not in o2
    assert "[read-coverage summary: solution.py]" in o2
    assert "requested lines: 5-14" in o2
    assert "already covered ranges: 1-40" in o2
    assert "# line 5" not in o2
    assert _READ_OVERLAP_COACHING not in o2, (
        "coverage replacement must not also append the neutral marker"
    )
    # Forbidden phrases (super loop integrity): no prescriptive language must leak into chat.
    forbidden = (
        "Do not re-read",
        "do not call `read`",
        "Do NOT call",
        "POST-TELEPORT",
        "workspace was swapped",
        "treat as the only source of truth",
    )
    for ph in forbidden:
        assert ph not in o2, f"silent intercept must not leak forbidden phrase: {ph!r}"
    # Counter increments for the redundant read path.
    assert ctx._redundant_read_count_by_path.get("solution.py") == 1

def test_record_tool_read_overlap_uses_symbol_coverage_raw_id(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "msym", "Draft", 200)
    ws = tmp_path / "wsym"
    ws.mkdir()
    sol = ws / "solution.py"
    lines = [
        "def foo(x):",
        '    """Foo transform."""',
        "    total = x",
        "    total += 1",
        "    total += 2",
        "    return total",
        "",
        "def bar(y):",
        "    value = y",
        "    return value",
    ]
    sol.write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=200,
        write_auto_snapshot_enabled=True,
        write_auto_snapshot_paths=("solution.py",),
    )

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = len(lines)
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[solution.py: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    first = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 1, "limit": 6},
        ToolResult(output=_fmt(1, 6)),
        raw_id="tool_000001_read.txt",
    )
    assert "[read-coverage summary:" not in first

    second = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 3, "limit": 2},
        ToolResult(output=_fmt(3, 2)),
        raw_id="tool_000002_read.txt",
    )
    assert "[read-coverage summary: solution.py::foo]" in second
    assert "previous full read raw_id: tool_000001_read.txt" in second
    assert "covered symbol range: 1-6" in second
    assert "total += 1" not in second

    lines[8] = "    value = y + 1"
    sol.write_text("\n".join(lines), encoding="utf-8")
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (10 lines, 150 bytes, sha256~def456)"),
    )
    third = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 4, "limit": 1},
        ToolResult(output=_fmt(4, 1)),
        raw_id="tool_000003_read.txt",
    )
    assert "[read-coverage summary: solution.py::foo]" in third
    assert "previous full read raw_id: tool_000001_read.txt" in third
    assert "total += 1" not in third

def test_record_tool_redundant_read_other_path_uses_neutral_marker(tmp_path) -> None:
    """Tier 2: redundant read on a non-snapshot path uses the neutral marker, no coaching."""
    mem = create_agent_memory(tmp_path / "mneut", "Draft", 200)
    ws = tmp_path / "wneut"
    ws.mkdir()
    p = ws / "other.txt"
    all_lines = [f"line{i}" for i in range(20)]
    p.write_text("\n".join(all_lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=200)

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 20
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = all_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[other.txt: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    ctx.record_tool_result(
        "read",
        {"path": "other.txt", "offset": 1, "limit": 20},
        ToolResult(output=_fmt(1, 20)),
    )
    o2 = ctx.record_tool_result(
        "read",
        {"path": "other.txt", "offset": 5, "limit": 5},
        ToolResult(output=_fmt(5, 5)),
    )
    # Non-snapshot path: keep body, append neutral marker (no canonical block).
    assert _AUTO_SNAPSHOT_PREFIX not in o2
    assert _READ_OVERLAP_COACHING in o2
    assert "Do not re-read" not in o2
    assert "Use that prior context" not in o2
