"""Deterministic adapter from LNR facts to ExecutionObservation."""

from __future__ import annotations

from typing import Any, Mapping

from scienceflow.foundation.contracts import ExecutionObservation


class ExecutionObservationBuilder:
    def from_lnr(
        self,
        *,
        command_id: str,
        elapsed_sec: float,
        signal: Mapping[str, Any],
        artifact_progress: bool,
        recoverable_artifact: bool,
    ) -> ExecutionObservation:
        raw_remaining = signal.get("deadline_remaining_sec")
        budget_known = raw_remaining is not None or bool(signal.get("deadline_event"))
        try:
            remaining = max(0.0, float(raw_remaining or 0.0))
        except (TypeError, ValueError):
            remaining = 0.0
            budget_known = False
        if not budget_known:
            # Unknown is deliberately not upgraded to a budget-exhausted signal.
            remaining = 1.0e12
        useful = signal.get("metric_value_useful")
        metric_improved = useful if isinstance(useful, bool) else None
        return ExecutionObservation(
            command_id=str(command_id),
            elapsed_sec=max(0.0, float(elapsed_sec or 0.0)),
            budget_remaining_sec=remaining,
            metric_improved=metric_improved,
            artifact_progress=bool(artifact_progress),
            recoverable_artifact=bool(recoverable_artifact),
            metadata={
                "source": "lnr_observer_shadow",
                "budget_known": budget_known,
                "legacy_metric_semantics": "resource_metric_value.useful",
            },
        )


__all__ = ["ExecutionObservationBuilder"]
