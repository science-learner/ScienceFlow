"""Contracts for hidden-path-aware agent-visible text projection."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from inquirycraft.tools import ToolResult

from scienceflow.agent.core.runtime.agent import ScienceAgent
from scienceflow.research.state.workspace.adapters.agent_visible_text import (
    HiddenPathPolicy,
    project_agent_visible_text,
    project_tool_output,
    rewrite_agent_visible_paths,
)
from scienceflow.research.state.workspace.adapters.path_hygiene import (
    _maybe_rewrite_tool_result_paths,
    _text_mentions_hidden_workspace_prefix,
    _tool_request_mentions_hidden_workspace_file,
)


def _agent(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _workspace_dir=tmp_path,
        _workspace_relative_path_mode=True,
        _teleport_mode="off",
        _agent_hidden_workspace_filenames=(".run_results.md",),
        _agent_hidden_workspace_path_prefixes=(
            ".run_results.md",
            "logs",
            "submission_snapshots",
            ".logs",
            ".agent_memory",
            ".git",
            ".memory",
            ".scienceflow_checkpoints",
        ),
    )


def test_prose_bare_logs_is_not_hidden(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    text = "writing verbose logs to workspace files"
    assert project_agent_visible_text(agent, text) == text
    assert rewrite_agent_visible_paths(agent, text) == text


def test_ls_a_omits_hidden_entries_without_placeholders(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    text = "[exit=0, 0.1s]\n.\n..\n.git\n.gitignore\n.logs\n.run_results.md\nREADME.md\nsolution.py"
    out = project_tool_output(
        agent,
        text,
        tool_name="bash",
        args={"command": "ls -a"},
    )
    lines = out.splitlines()
    assert ".git" not in lines
    assert ".logs" not in lines
    assert ".run_results.md" not in lines
    assert ".gitignore" in lines
    assert "README.md" in lines
    assert "solution.py" in lines
    assert "<hidden-control-path>" not in out


def test_ls_mixed_line_keeps_visible_entries(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    out = project_tool_output(
        agent,
        ".git, README.md, solution.py",
        tool_name="bash",
        args={"command": "ls -m"},
    )
    assert ".git" not in out
    assert "README.md" in out
    assert "solution.py" in out
    assert not out.lstrip().startswith(",")


def test_metric_line_keeps_metric_and_drops_hidden_path(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    out = project_tool_output(
        agent,
        "checkpoint saved to logs/model.pt final_score=0.91",
        tool_name="bash",
        args={"command": "python train.py"},
    )
    assert "logs/model.pt" not in out
    assert "final_score=0.91" in out
    assert "<hidden-control-path>" not in out


def test_url_paths_are_not_treated_as_workspace_hidden_paths(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    text = "see https://example.com/logs/model"
    out = project_tool_output(
        agent,
        text,
        tool_name="bash",
        args={"command": "python print_url.py"},
    )
    assert out == text


def test_read_output_preserves_code_lines_starting_with_bare_hidden_name(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    out = project_tool_output(
        agent,
        'logs = []\nrun_logs = []\nendpoint = "/logs"\nnormal = 1',
        tool_name="read",
        args={"path": "model.py"},
    )
    assert "logs = []" in out
    assert "run_logs = []" in out
    assert 'endpoint = "/logs"' in out
    assert "normal = 1" in out


def test_file_and_dir_listing_entries_are_removed_whole(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    out = project_tool_output(
        agent,
        "FILE .run_results.md (100 bytes)\nDIR logs/\nFILE submission.csv",
        tool_name="ls",
        args={"path": "."},
    )
    assert ".run_results.md" not in out
    assert "DIR logs" not in out
    assert "submission.csv" in out


def test_json_path_is_redacted_without_breaking_json(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    text = '{"path": ".git/config", "score": 0.9}'
    out = project_tool_output(
        agent,
        text,
        tool_name="bash",
        args={"command": "python -c 'print(...)'"},
    )
    assert ".git/config" not in out
    payload = json.loads(out)
    assert payload["path"] == ""
    assert payload["score"] == 0.9


def test_assistant_text_removes_hidden_paths_inline(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    out = project_agent_visible_text(
        agent,
        "checkpoint saved to logs/model.pt final_score=0.91",
    )
    assert "logs/model.pt" not in out
    assert "final_score=0.91" in out
    assert "<hidden-control-path>" not in out


def test_request_detection_still_recognizes_hidden_paths(tmp_path: Path) -> None:
    agent = _agent(tmp_path)

    assert _text_mentions_hidden_workspace_prefix(
        agent, "ls -la logs/ artifacts/", "logs"
    )
    assert _text_mentions_hidden_workspace_prefix(
        agent, "tail -30 .logs/train.log", ".logs"
    )
    assert not _text_mentions_hidden_workspace_prefix(
        agent, "tail -30 tmp/train_classifier.log", "logs"
    )

    assert _tool_request_mentions_hidden_workspace_file(
        agent, "bash", {"command": "ls -la submission_snapshots/ | head"}
    )
    assert _tool_request_mentions_hidden_workspace_file(
        agent, "read", {"path": ".run_results.md"}
    )
    assert _tool_request_mentions_hidden_workspace_file(
        agent, "read", {"path": str(tmp_path / "logs" / "train.log")}
    )
    assert not _tool_request_mentions_hidden_workspace_file(
        agent, "read", {"path": "artifacts/model.pt"}
    )


def test_feedback_pipeline_has_no_placeholder_after_rewrite(tmp_path: Path) -> None:
    dummy = _agent(tmp_path)
    dummy._tool_output_artifacts = None
    dummy._exec_feedback_max_chars = 4000
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, tool_name, args, tool_result, **kwargs):
            return tool_result.output

    dummy._memory_ctx = FakeMemoryCtx()

    result = ToolResult(
        output="[exit=0, 0.1s]\n.git\n.logs\n.run_results.md\nREADME.md\nsolution.py"
    )
    rewritten = _maybe_rewrite_tool_result_paths(dummy, result)

    # The rewrite stage must not mask hidden names; the feedback stage owns
    # the omission so metrics/log lines can be filtered with full context.
    assert ".git" in rewritten.output
    assert "<hidden-control-path>" not in rewritten.output

    feedback = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "ls -a"},
        rewritten,
    )
    assert "<hidden-control-path>" not in feedback
    assert ".git\n" not in f"{feedback}\n"
    assert ".logs\n" not in f"{feedback}\n"
    assert ".run_results.md\n" not in f"{feedback}\n"
    assert "README.md" in feedback
    assert "solution.py" in feedback


def test_hidden_request_coaching_is_redacted(tmp_path: Path) -> None:
    dummy = _agent(tmp_path)
    dummy._tool_output_artifacts = None
    dummy._exec_feedback_max_chars = 4000
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, tool_name, args, tool_result, **kwargs):
            raise AssertionError("hidden requests must not be recorded verbatim")

    dummy._memory_ctx = FakeMemoryCtx()

    feedback = dummy._prepare_tool_feedback_for_memory(
        "read",
        {"path": ".run_results.md"},
        ToolResult(output="secret stage ledger"),
        guard_coaching="see .git/config",
    )
    assert "hidden control file output omitted" in feedback
    assert ".git/config" not in feedback
    assert "<hidden-control-path>" not in feedback


def test_legacy_placeholders_are_removed_from_memory_text(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    out = project_agent_visible_text(
        agent,
        "old observation: <hidden-control-path> and <hidden-control-file>",
    )
    assert "<hidden-control-path>" not in out
    assert "<hidden-control-file>" not in out
    assert "old observation:" in out


def test_policy_from_agent_prefers_explicit_policy(tmp_path: Path) -> None:
    policy = HiddenPathPolicy(
        prefixes=(".control",),
        filenames=(".secret",),
    )
    agent = _agent(tmp_path)
    agent._hidden_path_policy = policy
    assert HiddenPathPolicy.from_agent(agent) is policy
