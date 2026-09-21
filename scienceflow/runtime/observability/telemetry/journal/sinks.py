"""Adapters from existing trace contracts to the correlation journal."""

from __future__ import annotations

from dataclasses import asdict

from scienceflow.runtime.core.kernel import HookTrace
from scienceflow.runtime.observability.telemetry.journal.correlation import CorrelationJournal


class HookTraceCorrelationSink:
    def __init__(self, journal: CorrelationJournal) -> None:
        self.journal = journal

    def __call__(self, trace: HookTrace) -> None:
        self.journal.record(
            source="runtime_hook",
            event_type="hook_trace",
            payload=asdict(trace),
            run_id=trace.run_id,
            worker_id=trace.worker_id,
            process_id=trace.process_id,
            stage_id=trace.stage_id,
            event_id=trace.event_id,
            sequence=trace.sequence,
        )


__all__ = ["HookTraceCorrelationSink"]
