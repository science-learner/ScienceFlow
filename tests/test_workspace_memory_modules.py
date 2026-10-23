# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Independent Workspace and Memory evolution contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from scienceflow.runtime.composition import build_lnr_module_graph
from scienceflow.foundation.contracts import MemoryView, StageCommitted, StageRecord
from scienceflow.research.state.knowledge.memory import MemoryService
from scienceflow.research.state.knowledge.memory.records.stage_memory import render_stage_cards as render_memory_cards
from scienceflow.research.solver.lnr.lifecycle.stage.records.stage_ledger import (
    StageCard,
    render_stage_cards as render_legacy_stage_cards,
)
from scienceflow.research.state.workspace import WorkspaceService, WorkspaceTransactionState


LEDGER = """# Run Results

### S01
metric: 2.4
lower_is_better: false
metric_validity: high
selection_eligible: true
BRIEF: geometric baseline
WHY: establishes a valid circle packing candidate
FILES: solution.py

### S02
metric: 2.5
lower_is_better: false
metric_validity: high
selection_eligible: true
BRIEF: refined placement
WHY: improves the accepted baseline
FILES: solution.py
"""


def test_workspace_recovers_stage_facts_without_memory_service(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ledger = workspace / "run_results.md"
    ledger.write_text(LEDGER, encoding="utf-8")
    service = WorkspaceService(
        root_dir=tmp_path,
        workspace_dir=workspace,
        ledger_path=ledger,
    )

    stages = service.list_stages()
    events = service.stage_committed_events()

    assert [stage.stage_id for stage in stages] == ["S01", "S02"]
    assert [event.sequence for event in events] == [1, 2]
    assert all(event.to_dict()["schema_version"] == "1.0" for event in events)
    assert len({event.event_id for event in events}) == 2


def test_memory_projection_is_idempotently_rebuilt_from_workspace_events(
    tmp_path: Path,
) -> None:
    stages = (
        StageRecord(stage_id="S01", metric="0.07", brief="baseline"),
        StageRecord(stage_id="S02", metric="0.06", brief="better validation"),
    )
    events = tuple(
        StageCommitted(
            event_id=f"event-{index}",
            workspace_id="nomad2018",
            stage=stage,
            sequence=index,
        )
        for index, stage in enumerate(stages, start=1)
    )
    memory = MemoryService(workspace_dir=tmp_path)

    first = memory.rebuild(events)
    assert memory.consume(events[-1]) is False
    second = memory.rebuild(reversed(events))

    assert first == second
    assert first.source_event_ids == ("event-1", "event-2")
    assert "S02: metric=0.06" in first.text


def test_memory_owned_stage_card_projection_preserves_legacy_prompt_bytes() -> None:
    cards = [
        StageCard(
            stage_id="S01",
            body="",
            metric="0.07",
            lower_is_better="true",
            metric_validity="medium",
            selection_eligible="true",
            metric_validity_reason_code="unknown_protocol",
            brief="baseline",
            why="first valid candidate",
            route_evidence="retain uncertainty",
            stage_events=("assessment complete",),
        )
    ]

    assert render_memory_cards(cards) == render_legacy_stage_cards(cards)


def test_memory_package_has_no_solver_dependency() -> None:
    root = Path(__file__).parents[1] / "scienceflow" / "research" / "state" / "memory"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module or "").startswith("scienceflow.research.solver"), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("scienceflow.research.solver")
                    for alias in node.names
                ), path


def test_workspace_owns_stage_transaction_prepare_and_rollback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ledger = workspace / "run_results.md"
    ledger.write_text(LEDGER, encoding="utf-8")
    service = WorkspaceService(
        root_dir=tmp_path,
        workspace_dir=workspace,
        ledger_path=ledger,
    )
    transactions = service.stage_transactions(
        transaction_root=tmp_path / "logs" / "stage_transactions"
    )
    after = LEDGER + "\n### S03\nmetric: 2.6\n"
    metric_event: dict[str, object] = {"artifact_sha": "artifact-3"}

    pending = transactions.prepare(
        transaction_id="L01-S03",
        stage_id="S03",
        lineage_id="L01",
        ledger_before=LEDGER,
        ledger_after=after,
        metric_event=metric_event,
    )
    ledger.write_text(after, encoding="utf-8")
    remaining, rolled_back = transactions.rollback(
        pending,
        stage_id="S03",
        reason="test_rollback",
    )

    manifest = json.loads(
        (tmp_path / "logs" / "stage_transactions" / "L01-S03.json").read_text()
    )
    assert remaining is None
    assert rolled_back is True
    assert ledger.read_text(encoding="utf-8") == LEDGER
    assert metric_event["stage_transaction_id"] == "L01-S03"
    assert manifest["status"] == "rolled_back"
    machine = transactions.machines[0]
    assert machine.state is WorkspaceTransactionState.ROLLED_BACK
    assert [row.current.value for row in machine.transitions] == [
        "prepared",
        "rolled_back",
    ]
    assert not hasattr(transactions, "memory")


def test_workspace_transaction_completion_uses_snapshot_identity(tmp_path: Path) -> None:
    ledger = tmp_path / "run_results.md"
    ledger.write_text(LEDGER, encoding="utf-8")
    service = WorkspaceService(
        root_dir=tmp_path,
        workspace_dir=tmp_path,
        ledger_path=ledger,
    )
    transactions = service.stage_transactions(transaction_root=tmp_path / "transactions")
    pending = transactions.prepare(
        transaction_id="L01-S03",
        stage_id="S03",
        lineage_id="L01",
        ledger_before=LEDGER,
        ledger_after=LEDGER,
        metric_event={},
    )

    remaining, committed = transactions.complete(
        pending,
        stage_id="S03",
        snapshot=SimpleNamespace(snapshot_id="snapshot-3", snapshot_path="snapshots/S03"),
    )

    assert remaining is None
    assert committed is True
    manifest = json.loads((tmp_path / "transactions" / "L01-S03.json").read_text())
    assert manifest["snapshot_id"] == "snapshot-3"
    assert manifest["status"] == "committed"
    assert transactions.machines[0].state is WorkspaceTransactionState.COMMITTED

    duplicate_pending, duplicate = transactions.complete(
        pending,
        stage_id="S03",
        snapshot=SimpleNamespace(snapshot_id="snapshot-3", snapshot_path="snapshots/S03"),
    )
    assert duplicate_pending is pending
    assert duplicate is False
    assert len(transactions.machines[0].transitions) == 2


def test_composition_uses_one_evaluator_gate_pair_and_separate_stores(
    tmp_path: Path,
) -> None:
    graph = build_lnr_module_graph(
        root_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        ledger_path=tmp_path / "workspace" / "run_results.md",
    )
    assert graph.stage_lifecycle.machine_for("S01").machine_type == "stage_lifecycle"
    assert graph.estra_planner.service is graph.estra
    assert graph.estra_archive.path.name == "decisions.jsonl"
    assert graph.assessment.evaluator_manager is graph.evaluator
    assert graph.assessment.gate_manager is graph.gate
    assert graph.estra is not None
    assert graph.resource_management is not None
    assert graph.admission is not None
    assert graph.execution_value is not None
    assert graph.finalization is not None
    assert graph.prompt_context is not None
    assert graph.telemetry.path.name == "correlation.jsonl"
    assert graph.workspace.workspace_dir == graph.memory.workspace_dir
    assert graph.workspace is not graph.memory


def test_workspace_and_memory_services_do_not_import_each_other() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    forbidden = {
        root / "research" / "state" / "workspace" / "session" / "service.py": (
            "scienceflow.research.state.knowledge.memory",
            "scienceflow.research.solver",
        ),
        root
        / "research"
        / "state"
        / "knowledge"
        / "memory"
        / "core"
        / "service.py": "scienceflow.research.state.workspace",
    }
    for path, prefix in forbidden.items():
        prefixes = (prefix,) if isinstance(prefix, str) else prefix
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module or "").startswith(prefixes), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(prefixes) for alias in node.names
                ), path


def test_new_workspace_memory_contracts_are_versioned() -> None:
    assert StageRecord.contract_schema_version() == "1.0"
    assert StageCommitted.contract_schema_version() == "1.0"
    assert MemoryView.contract_schema_version() == "1.0"
