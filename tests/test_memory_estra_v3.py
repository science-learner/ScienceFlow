# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Phase V3-7 Memory/EStra ownership and replay contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scienceflow.foundation.contracts import EstraContext
from scienceflow.research.control.estra import (
    EstraArchiveStore,
    EstraCommandKind,
    EstraDecisionEnvelope,
    EstraPlanRequest,
    EstraPlanner,
    map_estra_runtime_command,
)
from scienceflow.research.state.knowledge.memory import (
    MemoryCompactionRequest,
    MemoryCompactor,
    ProtectedContextAdapter,
    ProtectedContextRequest,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.estra import adapter as estra_adapter


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    text = "".join(json.dumps(record) + "\n" for record in records)
    (path / "short_term.json").write_text(text, encoding="utf-8")
    (path / "long_term.jsonl").write_text(text, encoding="utf-8")


def _record(role: str, content: str) -> dict[str, object]:
    return {
        "uuid": f"id-{role}-{content}",
        "message": {"__class__": "Message", "role": role, "content": content},
        "role": role,
        "extra_info": {},
        "timestamp": 1.0,
        "agent_id": "",
    }


def test_memory_compactor_strict_mode_preserves_only_task_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_records(
        source,
        [
            _record("system", "SYSTEM PREFIX"),
            _record("user", "FIRST TASK"),
            _record("assistant", "old exploration"),
            _record("tool", "old output"),
        ],
    )

    result = MemoryCompactor().compact(
        MemoryCompactionRequest(
            source_dir=str(source),
            destination_dir=str(destination),
            prompt="RESUME PROMPT",
            max_messages=100,
            strict_context_limit=True,
        )
    )

    written = (destination / "short_term.json").read_text(encoding="utf-8")
    assert result.applied is True
    assert result.record_count == 3
    assert result.source_record_count == 2
    assert result.compact_strength == "strict_context_limit"
    assert len(result.projection_hash) == 64
    assert "SYSTEM PREFIX" in written
    assert "FIRST TASK" in written
    assert "RESUME PROMPT" in written
    assert "old exploration" not in written


def test_memory_compactor_semantic_replay_hash_ignores_record_ids_and_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_records(source, [_record("user", "FIRST TASK")])
    compactor = MemoryCompactor()

    first = compactor.compact(
        MemoryCompactionRequest(
            source_dir=str(source),
            destination_dir=str(tmp_path / "first"),
            prompt="RESUME",
        )
    )
    second = compactor.compact(
        MemoryCompactionRequest(
            source_dir=str(source),
            destination_dir=str(tmp_path / "second"),
            prompt="RESUME",
        )
    )

    assert first.projection_hash == second.projection_hash


def test_memory_compactor_missing_source_is_a_typed_noop(tmp_path: Path) -> None:
    result = MemoryCompactor().compact(
        MemoryCompactionRequest(
            source_dir=str(tmp_path / "missing"),
            destination_dir=str(tmp_path / "destination"),
            prompt="RESUME",
        )
    )

    assert result.applied is False
    assert result.reason == "missing_source_memory"
    assert not (tmp_path / "destination").exists()


def test_protected_context_adapter_applies_prepared_summary_without_solver() -> None:
    calls: list[tuple[object, ...]] = []

    class Context:
        @staticmethod
        def replace_protected_raw_prefix_with_summary(
            end_index, summary, *, warn_chars, label
        ):
            calls.append((end_index, summary, warn_chars, label))
            return {"end_index": end_index, "chars": len(summary)}

    result = ProtectedContextAdapter().apply(
        Context(),
        ProtectedContextRequest(
            end_index=7,
            mode="facts",
            warn_chars=1000,
            summary="stable EDA facts",
            captured_before_commit=True,
        ),
    )

    assert result.applied is True
    assert result.protected_end_index == 7
    assert result.info["boundary_source"] == "pre_stage_commit"
    assert result.info["summary_chars"] == len("stable EDA facts")
    assert calls == [
        (7, "stable EDA facts", 1000, "LHR protected EDA facts")
    ]


def test_protected_context_adapter_reports_missing_capability() -> None:
    result = ProtectedContextAdapter().apply(
        object(),
        ProtectedContextRequest(end_index=4, mode="raw"),
    )

    assert result.applied is False
    assert result.reason == "capability_unavailable"


def test_protected_context_fact_projection_dedupes_noise_and_keeps_uncertainty() -> None:
    messages = [
        SimpleNamespace(role="user", content="task", tool_calls=[]),
        SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[{"name": "bash", "arguments": {"command": "inspect"}}],
        ),
        SimpleNamespace(
            role="tool",
            content=(
                "train shape: (100, 12)\n"
                "train shape: (100, 12)\n"
                "RandomForest n_estimators=500 validation RMSLE=0.4\n"
                "missing target values may require audit"
            ),
            tool_calls=[],
        ),
    ]

    summary = ProtectedContextAdapter.build_facts_summary(
        messages,
        end_index=3,
        max_chars=6000,
    )

    assert summary.count("train shape: (100, 12)") == 1
    assert "missing target values may require audit" in summary
    assert "RandomForest" not in summary
    assert "Removed raw pre-S01 scratch payloads: 1" in summary


@pytest.mark.asyncio
async def test_estra_planner_normalizes_and_hashes_model_envelope() -> None:
    request = EstraPlanRequest(
        request_id="estra-1",
        context=EstraContext(
            latest_stage="S03",
            switch_candidates=("S01", "S02"),
            trigger_source="context_limit",
        ),
        prompt="choose route",
    )
    planner = EstraPlanner()

    first = await planner.plan(
        request,
        lambda _: '{"action":"switch_stage","target_stage":"S01","reason":"best"}',
    )
    second = await planner.plan(
        request,
        lambda _: '{"action":"switch_stage","target_stage":"S01","reason":"best"}',
    )

    assert first.decision.action == "switch_stage"
    assert first.decision.target_stage == "S01"
    assert first.attempts[0].outcome == "valid"
    assert first.input_hash == second.input_hash
    assert first.output_hash == second.output_hash


@pytest.mark.asyncio
async def test_estra_planner_uses_isolated_fallback_after_invalid_primary() -> None:
    request = EstraPlanRequest(
        request_id="estra-2",
        context=EstraContext(
            latest_stage="S02",
            switch_candidates=("S01",),
            trigger_source="force_stage_capture",
        ),
        prompt="choose route",
    )

    result = await EstraPlanner().plan(
        request,
        lambda _: "<tool-call>invalid</tool-call>",
        primary_mode="main_agent_context",
        fallback=lambda _: '{"action":"keep_current","reason":"continue"}',
    )

    assert result.used_fallback is True
    assert result.decision_mode == "isolated_fallback"
    assert [attempt.outcome for attempt in result.attempts] == ["invalid", "valid"]
    assert result.decision.action == "keep_current"


@pytest.mark.asyncio
async def test_estra_planner_allows_redirect_without_historical_candidate() -> None:
    request = EstraPlanRequest(
        request_id="estra-deterministic",
        context=EstraContext(
            latest_stage="S01",
            switch_candidates=(),
            trigger_source="text_only",
        ),
        prompt="choose route",
    )

    result = await EstraPlanner().plan(
        request,
        lambda _: '{"action":"keep_but_redirect","reason":"try another approach"}',
    )

    assert result.decision_mode == "isolated"
    assert result.decision.action == "keep_but_redirect"
    assert result.decision.target_stage == "S01"
    assert result.decision.reason == "try another approach"
    assert result.attempts[0].mode == "isolated"


@pytest.mark.asyncio
async def test_estra_planner_error_falls_back_to_current_route() -> None:
    request = EstraPlanRequest(
        request_id="estra-error",
        context=EstraContext(
            latest_stage="S02",
            switch_candidates=("S01",),
            trigger_source="context_limit",
        ),
        prompt="choose route",
    )

    def fail(_):
        raise TimeoutError("controller timed out")

    result = await EstraPlanner().plan(request, fail)

    assert result.decision_mode == "safe_fallback"
    assert result.decision.action == "keep_current"
    assert result.decision.startpoint == "current_workspace"
    assert result.decision.intent == "continue"
    assert result.decision.target_stage == "S02"
    assert result.decision.compact is True


def test_estra_archive_is_idempotent_and_command_mapping_has_no_effect(
    tmp_path: Path,
) -> None:
    context = EstraContext(
        latest_stage="S02",
        switch_candidates=("S01",),
        trigger_source="manual",
    )
    decision = EstraPlanner().service.decide_from_text(
        '{"action":"switch_stage","target_stage":"S01","reason":"best"}',
        context=context,
    )
    assert decision is not None
    envelope = EstraDecisionEnvelope(
        envelope_id="envelope-1",
        request_id="estra-3",
        context=context,
        decision=decision,
        decision_mode="isolated",
        input_hash="input",
        output_hash="output",
    )
    store = EstraArchiveStore(tmp_path / "estra" / "decisions.jsonl")

    assert store.append(envelope) is True
    assert store.append(envelope) is False
    assert len(store.replay()) == 1
    command = map_estra_runtime_command(
        decision,
        command_id="command-1",
        expected_stage="S02",
        evidence_refs=("envelope-1",),
    )
    assert command.kind is EstraCommandKind.RESTORE_STAGE
    assert command.target_stage == "S01"
    assert command.expected_stage == "S02"
    assert command.evidence_refs == ("envelope-1",)

    recorded_store = EstraArchiveStore(tmp_path / "estra" / "recorded.jsonl")
    recorded = recorded_store.record(
        request_id="estra-recorded",
        context=context,
        decision=decision,
        decision_mode="isolated",
        raw='{"action":"switch_stage","target_stage":"S01"}',
        expected_stage="S02",
    )
    replayed = recorded_store.replay()[0]
    assert recorded.archived is True
    assert recorded.command.kind is EstraCommandKind.RESTORE_STAGE
    assert replayed["metadata"]["runtime_command"]["kind"] == "restore_stage"


def test_legacy_estra_emitter_delegates_to_replay_archive(tmp_path: Path) -> None:
    owner = SimpleNamespace(
        worker_id="W00",
        estra_archive_store=EstraArchiveStore(
            tmp_path / "estra" / "decisions.jsonl"
        ),
        _jsonl=lambda *_args, **_kwargs: None,
        _estra_axes_from_action=estra_adapter._estra_axes_from_action,
        _estra_decision_kind=estra_adapter._estra_decision_kind,
    )

    estra_adapter._emit_estra_decision(
        owner,
        action="switch_stage",
        target_stage="S01",
        latest_stage="S02",
        trigger_source="context_limit",
        compact=True,
        reason="restore stronger evidence",
        decision_mode="isolated",
        candidates=["S01", "S02"],
        switch_candidates=["S01"],
        candidate_node_uids={"S01": "W00:L01:S01", "S02": "W00:L01:S02"},
        raw='{"action":"switch_stage","target_stage":"S01"}',
    )

    row = owner.estra_archive_store.replay()[0]
    assert row["decision"]["target_stage"] == "S01"
    assert row["metadata"]["runtime_command"]["expected_stage"] == "S02"
