from __future__ import annotations

import ast
import json
from pathlib import Path

from scienceflow.research.control.admission import (
    AdmissionDecisionRouter,
    AdmissionObservationBuilder,
    AdmissionPolicyService,
    AdmissionReplayArchive,
)
from scienceflow.research.control.admission.policy import normalize_admission_decision
from scienceflow.foundation.config.schema.settings import LnrConfig
from scienceflow.research.control.execution_value import (
    ExecutionObservationBuilder,
    ExecutionValueDecisionRouter,
    ExecutionValueReplayArchive,
    ExecutionValueService,
)
from scienceflow.research.control.execution_value.adapters import LnrExecutionValueShadow


def _task_card() -> dict:
    return {
        "job_id": "job-1",
        "candidate_physical_gpus": ["0"],
        "admission_opportunity": {"lease_grantable_by_llm": True},
    }


def _blocked_rule() -> dict:
    return {
        "enabled": True,
        "acquired": False,
        "status": "PENDING",
        "admission_action": "PENDING",
        "reason": "gpu_busy",
        "requires_admission_review": True,
        "lease_grantable_by_llm": True,
        "candidate_physical_gpus": ["0"],
    }


def test_admission_observation_is_stable_and_detached_from_input() -> None:
    builder = AdmissionObservationBuilder()
    card = _task_card()
    rule = _blocked_rule()
    first = builder.build(task_card=card, rule_result=rule, mode="blocked_states")
    second = builder.build(task_card=card, rule_result=rule, mode="blocked_states")

    card["candidate_physical_gpus"].append("9")
    rule["reason"] = "mutated"

    assert first == second
    assert first.input_hash == second.input_hash
    assert first.observation_id.startswith("adobs_")
    assert first.task_card["candidate_physical_gpus"] == ["0"]
    assert first.rule_result["reason"] == "gpu_busy"


def test_admission_policy_is_effect_free_and_fails_closed_without_lease_path() -> None:
    decision = normalize_admission_decision(
        {"action": "RUN_NOW", "gpu_ids": ["0"], "reason": "start"},
        task_card={"job_id": "job-1"},
        rule_result={"status": "PENDING", "acquired": False},
        source="component",
    )

    assert decision["admission_action"] == "PENDING"
    assert decision["lease_grantable_by_llm"] is False
    service = AdmissionPolicyService()
    assert not hasattr(service, "acquire")
    assert not hasattr(service, "release")
    assert not hasattr(service, "kill")


def test_admission_router_records_shadow_diff_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "admission.jsonl"

    def legacy(raw, *, task_card, rule_result, source):
        out = normalize_admission_decision(
            raw,
            task_card=task_card,
            rule_result=rule_result,
            source=source,
        )
        return {**out, "confidence": "low"}

    archive = AdmissionReplayArchive(path)
    router = AdmissionDecisionRouter(
        source="component",
        archive=archive,
        legacy_normalizer=legacy,
    )
    record = router.decide(
        {"action": "RUN_NOW", "gpu_ids": ["0"], "confidence": "high"},
        task_card=_task_card(),
        rule_result=_blocked_rule(),
        mode="blocked_states",
        source="test",
    )
    replay = router.decide(
        {"action": "RUN_NOW", "gpu_ids": ["0"], "confidence": "high"},
        task_card=_task_card(),
        rule_result=_blocked_rule(),
        mode="blocked_states",
        source="test",
    )

    assert record.selected_source == "component"
    assert record.selected["admission_action"] == "RUN_NOW"
    assert record.safety_veto_preserved is True
    assert record.diff.equal is False
    assert record.diff.changed_fields == ("confidence",)
    assert replay.record_id == record.record_id
    assert len(path.read_text().splitlines()) == 1
    reopened = AdmissionReplayArchive(path)
    assert reopened.get(record.record_id)["selected_source"] == "component"


def test_admission_shadow_source_is_run_level_rollback() -> None:
    def legacy(raw, *, task_card, rule_result, source):
        out = normalize_admission_decision(
            raw,
            task_card=task_card,
            rule_result=rule_result,
            source=source,
        )
        return {**out, "admission_action": "PENDING", "action": "PENDING", "status": "PENDING"}

    record = AdmissionDecisionRouter(
        source="shadow",
        legacy_normalizer=legacy,
    ).decide(
        {"action": "RUN_NOW", "gpu_ids": ["0"]},
        task_card=_task_card(),
        rule_result=_blocked_rule(),
        mode="blocked_states",
        source="test",
    )

    assert record.selected_source == "legacy"
    assert record.selected["admission_action"] == "PENDING"
    assert record.component["admission_action"] == "RUN_NOW"


def test_admission_review_predicate_uses_selected_run_source() -> None:
    class ComponentPolicy(AdmissionPolicyService):
        def should_review(self, rule_result, *, mode):
            return False

    component = AdmissionDecisionRouter(source="component", service=ComponentPolicy())
    shadow = AdmissionDecisionRouter(source="shadow", service=ComponentPolicy())

    assert component.should_review(_blocked_rule(), mode="blocked_states") is False
    assert shadow.should_review(_blocked_rule(), mode="blocked_states") is True


def test_execution_observation_does_not_upgrade_unknown_metric_or_budget() -> None:
    observation = ExecutionObservationBuilder().from_lnr(
        command_id="train",
        elapsed_sec=120,
        signal={
            "resource_metric_value": {
                "validity": "medium",
                "reason": "unknown_protocol",
            }
        },
        artifact_progress=False,
        recoverable_artifact=True,
    )

    assert observation.metric_improved is None
    assert observation.budget_remaining_sec == 1.0e12
    assert observation.metadata["budget_known"] is False


def test_execution_value_router_archives_diff_and_safety_audit(tmp_path: Path) -> None:
    archive_path = tmp_path / "execution_value.jsonl"
    archive = ExecutionValueReplayArchive(archive_path)
    observation = ExecutionObservationBuilder().from_lnr(
        command_id="train",
        elapsed_sec=120,
        signal={"deadline_remaining_sec": 0.0, "deadline_event": True},
        artifact_progress=True,
        recoverable_artifact=True,
    )
    router = ExecutionValueDecisionRouter(
        source="component",
        service=ExecutionValueService(no_progress_timebox_sec=60),
        archive=archive,
    )
    selected, record = router.decide(
        observation,
        signal={"saw_final_score": True},
    )

    assert selected["action"] == "STOP"
    assert record.selected_source == "component"
    assert record.safety_audit.effect_free is True
    assert record.safety_audit.near_deliverable is True
    assert record.safety_audit.kill_intent_revalidation_required is True
    assert len(archive_path.read_text().splitlines()) == 1
    reopened = ExecutionValueReplayArchive(archive_path)
    assert reopened.get(record.record_id)["safety_audit"]["effect_free"] is True


def test_execution_value_default_component_preserves_adapter_payload() -> None:
    adapter = LnrExecutionValueShadow(
        ExecutionValueService(no_progress_timebox_sec=60),
        decision_source="component",
    )
    decision = adapter.decide(
        command_id="train",
        elapsed_sec=120,
        signal={"metric_value_useful": False},
        artifact_progress=False,
        recoverable_artifact=True,
    )

    assert decision["action"] == "TIMEBOX"
    assert set(decision) == {
        "action",
        "reason_code",
        "assessment",
        "timebox_sec",
        "resource_request",
        "schema_version",
        "observation",
    }
    assert adapter.last_record.selected_source == "component"


def test_execution_value_shadow_source_selects_legacy_without_effects() -> None:
    adapter = LnrExecutionValueShadow(
        ExecutionValueService(no_progress_timebox_sec=60),
        decision_source="shadow",
    )
    decision = adapter.decide(
        command_id="train",
        elapsed_sec=120,
        signal={"metric_value_useful": False},
        artifact_progress=False,
        recoverable_artifact=False,
    )

    assert decision["action"] == "CONTINUE"
    assert adapter.last_record.component["action"] == "TIMEBOX"
    assert not hasattr(adapter, "release")
    assert not hasattr(adapter, "kill")


def test_execution_value_output_is_never_consumed_as_resource_effect() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    observer_root = (
        root / "research" / "solver" / "lnr" / "resources" / "runtime" / "observer"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in observer_root.rglob("*.py")
    )

    # The adapter writes one advisory field; resource kill/release paths do not read it.
    assert source.count("job.execution_value_shadow") == 1
    assert "job.execution_value_shadow = self.execution_value_shadow.decide(" in source
    assert (
        'Path(payload["task_resource_dir"]) / "module_state" / target.worker_id.lower()'
        in source
    )


def test_admission_and_execution_value_dependency_boundaries() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    forbidden = {
        root / "research" / "control" / "admission": (
            "scienceflow.research.control.resources.policy.effects",
            "scienceflow.research.control.resources.policy.mechanisms",
            "scienceflow.research.control.resources.runtime.service",
            "scienceflow.research.solver",
        ),
        root / "research" / "control" / "execution_value": (
            "scienceflow.research.control.resources",
            "scienceflow.research.solver",
        ),
    }
    for directory, prefixes in forbidden.items():
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not str(node.module or "").startswith(prefixes), path
                elif isinstance(node, ast.Import):
                    assert all(not alias.name.startswith(prefixes) for alias in node.names), path


def test_run_level_decision_source_defaults_are_explicit() -> None:
    config = LnrConfig()
    assert config.resource_admission_decision_source == "component"
    assert config.resource_execution_value_decision_source == "component"


def test_legacy_admission_import_path_is_identity_compatible() -> None:
    from scienceflow.research.control.admission.policy import should_request_admission_llm as canonical
    from scienceflow.research.solver.lnr.resources.runtime.control.admission.admission import (
        should_request_admission_llm as legacy,
    )

    assert legacy is canonical


def test_replay_records_are_valid_json_lines(tmp_path: Path) -> None:
    archive = AdmissionReplayArchive(tmp_path / "admission.jsonl")
    record = AdmissionDecisionRouter(source="component", archive=archive).decide(
        {"action": "PENDING"},
        task_card=_task_card(),
        rule_result=_blocked_rule(),
        mode="blocked_states",
        source="test",
    )
    row = json.loads((tmp_path / "admission.jsonl").read_text())
    assert row["record_id"] == record.record_id
    assert row["schema_version"] == "1.0"
