#!/usr/bin/env python3
"""Capture and compare the deterministic ScienceFlow Tool Runtime contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from inquirycraft.tools import classify_tool_call_bundle

from scienceflow.runtime.safety.policy.agent_policies.bash_command_classifier import (
    classify_bash_command,
)
from scienceflow.runtime.safety.policy.agent_policies.bash_utils import (
    _bash_command_parallel_safe,
)
from scienceflow.runtime.safety.execution.agent_runtime.tool_composition import create_tool_collection
from scienceflow.runtime.safety.tooling.workspace.shell_guards import (
    background_resource_command_blocked_error,
    dangerous_delete_command_blocked_error,
    global_filesystem_scan_blocked_error,
    interactive_stdin_blocked_error,
    normalize_bash_command_for_agent,
    privilege_escalation_blocked_error,
    process_control_blocked_error,
    shared_python_env_write_blocked_error,
    workspace_scope_path_blocked_error,
)
from scienceflow.runtime.safety.tooling.workspace.shell_output import _dedup_repeated_blocks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    ROOT / "tests" / "fixtures" / "tool_runtime_contract" / "baseline_v1.json"
)


def _result(value: Any) -> dict[str, Any]:
    return {
        "type": type(value).__name__,
        "output": getattr(value, "output", None),
        "error": getattr(value, "error", None),
        "system": getattr(value, "system", None),
        "rendered": str(value),
        "truthy": bool(value),
    }


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        ),
    )


def _bundle_modes() -> dict[str, str]:
    read = _tool_call("c-read", "read", {"path": "notes/example.txt"})
    grep = _tool_call("c-grep", "grep", {"pattern": "alpha"})
    write = _tool_call("c-write", "write", {"path": "x", "content": "x"})
    bash_a = _tool_call("c-bash-a", "bash", {"command": "python3 a.py"})
    bash_b = _tool_call("c-bash-b", "bash", {"command": "python3 b.py"})
    common = {
        "readonly_tool_names": {"read", "grep", "glob", "ls"},
        "mutation_tool_names": {"write", "edit"},
        "parallel_bash_enabled": True,
        "bash_parallel_safe": _bash_command_parallel_safe,
    }
    return {
        "single": classify_tool_call_bundle([read], **common),
        "readonly": classify_tool_call_bundle([read, grep], **common),
        "mutation": classify_tool_call_bundle([read, write], **common),
        "parallel_bash": classify_tool_call_bundle([bash_a, bash_b], **common),
        "forced_sequential": classify_tool_call_bundle(
            [read, grep],
            force_sequential=True,
            **common,
        ),
    }


async def capture_tool_runtime_contract(workspace: Path) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    tools = create_tool_collection(workspace)
    schema = tools.to_params()
    schema_bytes = json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode()

    written = await tools.execute(
        name="write",
        tool_input={"path": "notes/example.txt", "content": "alpha\nbeta\n"},
    )
    read_before = await tools.execute(
        name="read",
        tool_input={"path": "notes/example.txt", "offset": 1, "limit": 200},
        config={"ignored": True},
    )
    edited = await tools.execute(
        name="edit",
        tool_input={
            "path": "notes/example.txt",
            "old_str": "beta",
            "new_str": "gamma",
        },
    )
    read_after = await tools.execute(
        name="read",
        tool_input={"path": "notes/example.txt", "offset": 1, "limit": 200},
    )
    grep = await tools.execute(
        name="grep",
        tool_input={"pattern": "alpha|gamma", "path": "notes", "include": "*.txt"},
    )
    glob = await tools.execute(
        name="glob",
        tool_input={"pattern": "**/*.txt", "path": "."},
    )
    listed = await tools.execute(
        name="ls",
        tool_input={"path": ".", "recursive": True, "depth": 2},
    )

    invalid = await tools.execute(name="missing", tool_input={})
    bash_empty = await tools.execute(name="bash", tool_input={"command": "  "})
    write_missing_path = await tools.execute(
        name="write",
        tool_input={"content": "body"},
    )
    write_missing_content = await tools.execute(
        name="write",
        tool_input={"path": "missing.txt"},
    )
    write_placeholder = await tools.execute(
        name="write",
        tool_input={
            "path": "placeholder.txt",
            "content": "<<<WRITE_PLACEHOLDER: 100 bytes>>>",
        },
    )
    read_escape = await tools.execute(
        name="read",
        tool_input={"path": "../outside.txt"},
    )
    edit_missing = await tools.execute(
        name="edit",
        tool_input={"path": "notes/example.txt", "old_str": "missing", "new_str": "x"},
    )

    command_samples = [
        "",
        "cat > x.py <<'PY'\nprint(1)\nPY",
        "cp a b",
        "mv a b",
        "pip install numpy",
        "git status",
        "python3 solution.py",
        "ls -la",
        "python3 train.py",
        "echo ok",
    ]
    normalized, pip_rewritten = normalize_bash_command_for_agent(
        "python -m pip install x"
    )

    return {
        "schema": {
            "names": [item["function"]["name"] for item in schema],
            "bytes": len(schema_bytes),
            "sha256": hashlib.sha256(schema_bytes).hexdigest(),
        },
        "context": {
            "write": _result(written),
            "read_before": _result(read_before),
            "edit": _result(edited),
            "read_after": _result(read_after),
            "grep": _result(grep),
            "glob": _result(glob),
            "ls": _result(listed),
        },
        "mechanism": {
            "bundle_modes": _bundle_modes(),
            "command_kinds": {
                command: classify_bash_command(command) for command in command_samples
            },
            "normalization": {"command": normalized, "pip_rewritten": pip_rewritten},
            "parallel_safe": {
                "read": _bash_command_parallel_safe("python3 inspect.py"),
                "write": _bash_command_parallel_safe(
                    "cat > result.txt <<'EOF'\nx\nEOF"
                ),
            },
            "dedup": _dedup_repeated_blocks("a\na\na\nb\n", min_repeat=3),
        },
        "error_text": {
            "invalid": _result(invalid),
            "bash_empty": _result(bash_empty),
            "write_missing_path": _result(write_missing_path),
            "write_missing_content": _result(write_missing_content),
            "write_placeholder": _result(write_placeholder),
            "read_escape": _result(read_escape),
            "edit_missing": _result(edit_missing),
            "guards": {
                "background": background_resource_command_blocked_error(
                    "python3 train.py > train.log 2>&1 &"
                ),
                "delete": dangerous_delete_command_blocked_error("rm -rf outputs"),
                "global_scan": global_filesystem_scan_blocked_error(
                    "find / -name '*.csv'"
                ),
                "workspace_scope": workspace_scope_path_blocked_error(
                    "cat /tmp/data.txt",
                    workspace,
                ),
                "shared_env": shared_python_env_write_blocked_error(
                    "pip install numpy"
                ),
                "privilege": privilege_escalation_blocked_error("sudo apt-get update"),
                "process": process_control_blocked_error("pkill -f python"),
                "interactive": interactive_stdin_blocked_error("vim result.txt"),
            },
        },
        "workspace": {
            "files": [
                str(path.relative_to(workspace))
                for path in sorted(workspace.rglob("*"))
                if path.is_file()
            ],
            "example_text": (workspace / "notes" / "example.txt").read_text(
                encoding="utf-8"
            ),
            "example_sha256": hashlib.sha256(
                (workspace / "notes" / "example.txt").read_bytes()
            ).hexdigest(),
        },
    }


def compare_contract(
    current: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    categories = ("schema", "context", "mechanism", "error_text", "workspace")
    expected_hashes = expected["category_sha256"]
    current_hashes = {
        category: hashlib.sha256(
            json.dumps(
                current.get(category),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for category in categories
    }
    diffs = [
        category
        for category in categories
        if current_hashes[category] != expected_hashes.get(category)
    ]
    return {
        "benchmark": "tool_runtime_contract",
        "baseline": "baseline_v1",
        "totals": {
            "schema_diff_count": int("schema" in diffs),
            "context_diff_count": int("context" in diffs),
            "mechanism_diff_count": int("mechanism" in diffs),
            "error_text_diff_count": int("error_text" in diffs),
            "workspace_diff_count": int("workspace" in diffs),
            "failed_categories": len(diffs),
        },
        "failed_categories": diffs,
        "category_sha256": current_hashes,
        "verdict": "pass" if not diffs else "fail",
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-output", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="scienceflow-tool-contract-") as tmp:
        current = await capture_tool_runtime_contract(Path(tmp))
    if args.capture_output:
        args.capture_output.parent.mkdir(parents=True, exist_ok=True)
        args.capture_output.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = compare_contract(current, expected)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
