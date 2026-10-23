"""Fail-closed probe for the existing GateManager during onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scienceflow.foundation.contracts import EvalContext, MetricEvent
from scienceflow.research.quality.gate.policy import GateManager


def run_gate_probe(
    *,
    task_id: str,
    workspace: Path,
    evaluator: dict[str, Any],
    gate: dict[str, Any],
    backend_name: str,
    artifact_path: str,
    metric_name: str,
    lower_is_better: bool | None,
) -> tuple[bool, str]:
    context = EvalContext(
        task_profile=str(evaluator.get("task_profile") or "auto"),
        task_id=task_id,
        task_root=workspace,
        workspace=workspace,
        worker_id="preflight",
        cfg={"evaluator": evaluator, "gate": gate},
    )
    common = {
        "candidate_id": "preflight-probe",
        "worker_id": "preflight",
        "stage_id": "preflight",
        "metric_name": metric_name or "metric",
        "lower_is_better": lower_is_better,
        "artifact_path": artifact_path,
        "artifact_sha": "preflight",
        "evaluator_backend": backend_name,
        "metric_type": str((evaluator.get("metric") or {}).get("type") or "benchmark"),
    }
    failed_event = MetricEvent(
        **common,
        metric_value=None,
        validation_ok=False,
        candidate_ready=False,
        selection_eligible=False,
        metric_validity="low",
        metric_validity_reason_code="preflight_failure_probe",
        evaluator_status="preflight_failure_probe",
    )
    valid_event = MetricEvent(
        **common,
        metric_value=1.0,
        validation_ok=True,
        candidate_ready=True,
        selection_eligible=True,
        metric_validity="high",
        metric_validity_reason_code="preflight_valid_probe",
        evaluator_status="success",
    )
    manager = GateManager.default()
    rejected = manager.decide(context, failed_event, trigger="final")
    accepted = manager.decide(context, valid_event, trigger="final")
    ok = not rejected.accepted and rejected.action == "reject" and accepted.accepted
    detail = (
        "configured policy accepts a valid probe and rejects evaluator failure"
        if ok
        else f"valid={accepted.reason_code}; failure={rejected.reason_code}"
    )
    return ok, detail


__all__ = ["run_gate_probe"]
