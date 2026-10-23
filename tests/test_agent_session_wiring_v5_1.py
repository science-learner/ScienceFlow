from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from inquirycraft.memory import Memory, Message
from inquirycraft.runtime import ToolCommitObservation
from inquirycraft.tools import ToolResult
from scienceflow.agent import ScienceAgent
from scienceflow.agent.core.ports.callback_ports import (
    AgentCallbackPorts,
    install_callback_ports,
)
from scienceflow.agent.core.ports.session_callbacks import ScienceFlowPostCommitAdapter
from scienceflow.agent.session import _deadline_capped_llm_timeout
from scienceflow.agent.core.runtime.run_policy import RoundContext
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    _WallClockAutoContinuePolicy,
)

from tests._science_agent_repl_support import _FakeLLM, _tool_msg


@pytest.mark.asyncio
async def test_real_session_reaches_callbacks_and_persists_correlated_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "solution.py").write_text(
        "print('Best Validation wAUC: 0.641136')\n", encoding="utf-8"
    )
    calls: list[str] = []

    def archive(**kwargs):
        calls.append("candidate_archive")
        assert "0.641136" in str(kwargs["tool_result"].system)

    def metric(**_):
        calls.append("metric_interpretation")
        return {
            "metric_found": True,
            "metric_name": "weighted_auc",
            "metric_value": 0.641136,
            "split": "validation",
            "is_final": True,
            "evidence_line": "Best Validation wAUC: 0.641136",
            "confidence": "high",
        }

    async def stage(**_):
        calls.append("stage_capture")
        return "STAGE_FINAL"

    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                _tool_msg("bash", {"command": "python3 solution.py"}),
                SimpleNamespace(tool_calls=[], content="UNEXPECTED_SECOND_TURN"),
            ]
        ),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._scienceflow_worker_id = "W07"
    agent._scienceflow_task_profile = "mlebench"
    agent._embedded_full_run_enabled = False
    install_callback_ports(
        agent,
        AgentCallbackPorts(
            candidate_archive=archive,
            metric_interpretation=metric,
            stage_capture=stage,
        ),
    )
    monkeypatch.setenv("SCIENCEFLOW_RUN_ID", "run-v5-1")

    assert await agent.run("evaluate") == "STAGE_FINAL"
    assert calls == ["candidate_archive", "metric_interpretation", "stage_capture"]

    events_path = tmp_path / ".logs" / "agent_runtime_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    types = [row["type"] for row in events]
    assert types.index("tool.completed") < types.index("tool.commit.observed")
    assert {row["run_id"] for row in events} == {"run-v5-1"}
    assert {row["agent_id"] for row in events} == {"W07"}
    assert all(
        row["payload"]["correlation"]["stage_id"] == "draft" for row in events[:2]
    )
    session_ids = {row["session_id"] for row in events}
    assert len(session_ids) == 1
    assert next(iter(session_ids)).startswith("run-v5-1-W07-")

    journals = list((tmp_path / ".logs" / "agent_runtime_operations").glob("*.jsonl"))
    assert len(journals) == 1
    operations = [json.loads(line) for line in journals[0].read_text().splitlines()]
    settled = next(
        row for row in operations if row["kind"] == "tool" and row["phase"] == "settled"
    )
    assert settled["payload"]["message"]["role"] == "tool"


@pytest.mark.asyncio
async def test_text_only_callback_precedes_generic_text_policy(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=_FakeLLM([SimpleNamespace(tool_calls=[], content="candidate ready")]),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    install_callback_ports(
        agent,
        AgentCallbackPorts(text_only_decision=lambda **_: "TEXT_FINAL"),
    )

    assert await agent.run("finish") == "TEXT_FINAL"
    provider = json.loads(
        (tmp_path / ".logs" / "agent_provider_calls.jsonl").read_text().splitlines()[0]
    )
    assert provider["usage_status"] == "unknown"
    assert provider["cache_rate"] is None


@pytest.mark.asyncio
async def test_unparsed_tool_markup_retries_inside_one_session(tmp_path: Path) -> None:
    malformed = '→ bash {"command": </parameter> </function> </tool_call>'
    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                SimpleNamespace(tool_calls=[], content=malformed),
                SimpleNamespace(
                    tool_calls=[], content="No candidate could be produced."
                ),
            ]
        ),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )

    assert await agent.run("solve") == "No candidate could be produced."
    assert agent._scienceflow_worker_search_outcome == "stop_no_candidate"
    provider = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(provider) == 2
    assert len({row["session_id"] for row in provider}) == 1
    assert agent._scienceflow_unparsed_tool_call_retries == 1


@pytest.mark.parametrize(
    "malformed",
    [
        '→ bash {"command": </parameter> </function> </tool_call>',
        '→ bash {"command": "python3 solution.py"}',
    ],
)
@pytest.mark.asyncio
async def test_unparsed_tool_markup_recovers_before_text_only_callback(
    tmp_path: Path, malformed: str
) -> None:
    callback_calls = 0

    def text_only_callback(**_kwargs):
        nonlocal callback_calls
        callback_calls += 1
        return "TEXT_FINAL"

    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                SimpleNamespace(tool_calls=[], content=malformed),
                SimpleNamespace(tool_calls=[], content="candidate ready"),
            ]
        ),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    install_callback_ports(
        agent,
        AgentCallbackPorts(text_only_decision=text_only_callback),
    )

    assert await agent.run("solve") == "TEXT_FINAL"
    assert callback_calls == 1
    provider = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(provider) == 2
    assert len({row["session_id"] for row in provider}) == 1
    assert agent._scienceflow_unparsed_tool_call_retries == 1


def test_wall_clock_policy_never_stops_text_only_before_deadline(
    tmp_path: Path,
) -> None:
    policy = _WallClockAutoContinuePolicy(
        deadline_monotonic=time.monotonic() + 60,
        max_text_only_retries=2,
    )
    ctx = RoundContext(0, 500, "status only", tmp_path)

    decisions = [policy.on_text_only(ctx) for _ in range(6)]

    assert all(should_continue for should_continue, _ in decisions)
    assert any("LNR_CONTINUE_SEARCH" in str(prompt) for _, prompt in decisions)


def test_provider_timeout_preserves_deadline_across_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = SimpleNamespace(
        _llm_stream_timeout_sec=600,
        _llm_tool_stream_max_attempts=2,
    )
    policy = SimpleNamespace(deadline_monotonic=1120.0)
    monkeypatch.setattr("scienceflow.agent.session.time.monotonic", lambda: 1000.0)

    assert _deadline_capped_llm_timeout(host, policy) == 60.0
    assert _deadline_capped_llm_timeout(host, SimpleNamespace()) == 600.0


@pytest.mark.asyncio
async def test_pending_stage_commit_callback_handles_unparsed_tool_markup(
    tmp_path: Path,
) -> None:
    malformed = '→ bash {"command": </parameter> </function> </tool_call>'
    callback_calls = 0

    def stage_commit_callback(**kwargs):
        nonlocal callback_calls
        callback_calls += 1
        kwargs["agent"]._lnr_stage_commit_text_pending = False
        kwargs["agent"]._lnr_stage_commit_text_handled = True
        kwargs["agent"]._lnr_transient_user_prompt = ""
        return None

    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                SimpleNamespace(tool_calls=[], content=malformed),
                SimpleNamespace(tool_calls=[], content="candidate committed"),
            ]
        ),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    agent._lnr_stage_commit_text_pending = True
    install_callback_ports(
        agent,
        AgentCallbackPorts(text_only_decision=stage_commit_callback),
    )

    assert await agent.run("commit stage") == "candidate committed"
    assert callback_calls == 2
    assert getattr(agent, "_scienceflow_unparsed_tool_call_retries", 0) == 0
    provider = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(provider) == 2
    assert len({row["session_id"] for row in provider}) == 1


@pytest.mark.asyncio
async def test_pending_stage_commit_callback_blocks_structured_tool_call(
    tmp_path: Path,
) -> None:
    callback_calls = 0

    def stage_commit_callback(**kwargs):
        nonlocal callback_calls
        callback_calls += 1
        kwargs["agent"]._lnr_stage_commit_text_pending = False
        kwargs["agent"]._lnr_stage_commit_text_handled = True
        kwargs["agent"]._lnr_transient_user_prompt = ""
        return None

    forbidden = tmp_path / "structured_tool_ran"
    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                _tool_msg("bash", {"command": f"touch {forbidden}"}),
                SimpleNamespace(tool_calls=[], content="candidate committed"),
            ]
        ),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    agent._lnr_stage_commit_text_pending = True
    install_callback_ports(
        agent,
        AgentCallbackPorts(text_only_decision=stage_commit_callback),
    )

    assert await agent.run("commit stage") == "candidate committed"
    assert callback_calls == 2
    assert not forbidden.exists()
    runtime_events = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_runtime_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert not any(row["type"] == "tool.started" for row in runtime_events)


@pytest.mark.asyncio
async def test_stage_commit_parse_retry_stays_inside_one_session(
    tmp_path: Path,
) -> None:
    agent = ScienceAgent(
        llm=_FakeLLM(
            [
                SimpleNamespace(tool_calls=[], content="missing stage block"),
                SimpleNamespace(tool_calls=[], content="valid stage block"),
            ]
        ),
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    attempts = 0

    def stage_text(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            kwargs["agent"]._lnr_stage_commit_text_handled = True
            kwargs["agent"]._lnr_transient_user_prompt = "retry stage commit"
            return None
        return "STAGE_FINAL"

    install_callback_ports(
        agent,
        AgentCallbackPorts(text_only_decision=stage_text),
    )

    assert await agent.run("commit stage") == "STAGE_FINAL"
    provider = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert attempts == 2
    assert len(provider) == 2
    assert len({row["session_id"] for row in provider}) == 1


@pytest.mark.asyncio
async def test_stable_runtime_audit_survives_workspace_log_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "worker" / "workspace"
    audit_dir = tmp_path / "worker" / "logs" / "agent_runtime_audit"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("SCIENCEFLOW_RUN_ID", "run-audit-rollover")

    for content in ("first bounded stop", "second bounded stop"):
        agent = ScienceAgent(
            llm=_FakeLLM([SimpleNamespace(tool_calls=[], content=content)]),
            memory=Memory(max_messages=20),
            workspace_dir=workspace,
            max_steps=1,
        )
        agent._scienceflow_worker_id = "W00"
        agent._scienceflow_runtime_log_dir = audit_dir
        assert await agent.run("solve") == content
        shutil.rmtree(workspace / ".logs", ignore_errors=True)

    provider = [
        json.loads(line)
        for line in (audit_dir / "agent_provider_calls.jsonl").read_text().splitlines()
    ]
    runtime = [
        json.loads(line)
        for line in (audit_dir / "agent_runtime_events.jsonl").read_text().splitlines()
    ]
    assert len(provider) == 2
    assert len({row["session_id"] for row in provider}) == 2
    assert sum(row["type"] == "session.started" for row in runtime) == 2
    assert [row["sequence"] for row in runtime] == list(range(1, len(runtime) + 1))


@pytest.mark.asyncio
async def test_context_early_result_skips_provider_and_completes_session(
    tmp_path: Path,
) -> None:
    phases: list[str] = []
    llm = _FakeLLM([])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._memory_ctx.build_messages_for_llm_with_stats = lambda: (
        [SimpleNamespace(role="user", content="visible")],
        1,
    )
    install_callback_ports(
        agent,
        AgentCallbackPorts(
            context_limit_estra=lambda **_: "CONTEXT_FINAL",
            context_compact_observer=lambda *, phase, **_: phases.append(phase),
        ),
    )

    assert await agent.run("finish from context") == "CONTEXT_FINAL"
    assert phases == ["estra_check", "deferred"]
    events = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_runtime_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "llm.requested" not in {row["type"] for row in events}
    assert events[-1]["type"] == "session.completed"


@pytest.mark.asyncio
async def test_wall_clock_policy_stops_before_next_provider_round(
    tmp_path: Path,
) -> None:
    llm = _FakeLLM([])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    agent._run_policy.on_round_start = lambda _context: False

    assert await agent.run("solve") == "[ScienceAgent] Wall-clock deadline reached."
    assert agent._scienceflow_worker_search_outcome == "stop_no_candidate"
    events = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_runtime_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "llm.requested" not in {row["type"] for row in events}
    assert events[-1]["type"] == "session.completed"


@pytest.mark.asyncio
async def test_settled_resume_does_not_repeat_scienceflow_post_commit_effects() -> None:
    calls: list[str] = []
    host = SimpleNamespace(
        _host_ports=SimpleNamespace(
            normalize_tool_input_paths=lambda _name, args: args,
        ),
        _log_info=lambda *_args: None,
    )
    install_callback_ports(
        host,
        AgentCallbackPorts(candidate_archive=lambda **_: calls.append("archive")),
    )
    adapter = ScienceFlowPostCommitAdapter(host)
    result = ToolResult(output="already committed")
    common = {
        "call": {"id": "call-resume"},
        "invocation": SimpleNamespace(
            id="call-resume", name="read", arguments={"path": "result.md"}
        ),
        "raw_result": result,
        "result": result,
        "message": Message(
            "tool", "already committed", name="read", tool_call_id="call-resume"
        ),
        "bundle_index": 0,
        "bundle_mode": "single",
        "journal_attempt": 1,
        "journal_sequence": 4,
        "settled_event_id": "event-settled",
    }
    context = SimpleNamespace(run_id="run-resume", metadata={})

    assert (
        await adapter(
            ToolCommitObservation(**common, recovering=True, replayed=False), context
        )
        is None
    )
    assert calls == []

    assert (
        await adapter(
            ToolCommitObservation(**common, recovering=True, replayed=True), context
        )
        is None
    )
    assert calls == ["archive"]


def test_stage_id_refresh_accepts_coordinator_mapping() -> None:
    class Owner:
        stage_snapshots = {
            "S01": SimpleNamespace(stage_id="S01"),
            "S02": SimpleNamespace(stage_id="S02"),
        }

        def callback(self):
            return None

    context = SimpleNamespace(
        metadata={"correlation": {"stage_id": "draft"}},
    )

    ScienceFlowPostCommitAdapter._refresh_stage_id(Owner().callback, context)

    assert context.metadata["stage_id"] == "S02"
    assert context.metadata["correlation"]["stage_id"] == "draft"


def test_stage_id_refresh_advances_active_stage_and_lineage() -> None:
    agent = SimpleNamespace()

    class Owner:
        stage_snapshots = {"S01": SimpleNamespace(stage_id="S01")}
        _resource_main_agent_ref = agent

        @staticmethod
        def _next_stage_id_for_logging() -> str:
            return "S02"

        @staticmethod
        def _lineage_uid_prefix() -> str:
            return "L03"

        @staticmethod
        def _stage_node_uid(stage_id: str, *, lineage_id: str) -> str:
            return f"W07:{lineage_id}:{stage_id}"

        def callback(self):
            return None

    owner = Owner()
    context = SimpleNamespace(
        metadata={"correlation": {"stage_id": "S01", "lineage_id": "L02"}},
    )

    ScienceFlowPostCommitAdapter._refresh_stage_id(owner.callback, context)

    assert context.metadata["stage_id"] == "S02"
    assert context.metadata["lineage_id"] == "L03"
    assert context.metadata["node_uid"] == "W07:L03:S02"
    assert context.metadata["correlation"] == {
        "stage_id": "S01",
        "lineage_id": "L02",
    }
    assert agent._scienceflow_stage_id == "S02"
    assert agent._scienceflow_lineage_id == "L03"
    assert agent._scienceflow_node_uid == "W07:L03:S02"


def test_pending_stage_blocks_duplicate_evaluator_in_same_tool_bundle(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "best_solution.json").write_text("{}\n", encoding="utf-8")
    host = SimpleNamespace(
        _lnr_candidate_artifact_rel="artifacts/best_solution.json",
        _lnr_stage_capture_on_candidate_artifact=True,
        _embedded_full_run_enabled=False,
        _lnr_snapshot_ok=False,
        _workspace_dir=tmp_path,
    )
    adapter = ScienceFlowPostCommitAdapter(host)

    class Owner:
        pending_text_stage_commit = {"stage_id": "S01"}

        def callback(self, **_kwargs):
            raise AssertionError("pending stage must suppress duplicate evaluation")

    assert adapter._stage_ready(ToolResult(output="ok"), Owner().callback) is False
