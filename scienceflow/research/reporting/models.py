"""Stable, renderer-neutral final report model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ReportDocument:
    schema_version: int
    generated_at_utc: str
    run_id: str
    attempt: int
    task_name: str
    status: str
    started_at: float | None
    finished_at: float | None
    elapsed_sec: float
    budget_sec: float
    metric_name: str
    lower_is_better: bool
    best: str
    evaluated: int
    valid: int
    workers: list[str] = field(default_factory=list)
    worker_counts: dict[str, int] = field(default_factory=dict)
    estra: dict[str, int] = field(default_factory=dict)
    eec: dict[str, int] = field(default_factory=dict)
    eec_total: int = 0
    gpu: str = "—"
    usage: str = ""
    cost: str = "—"
    failure_kind: str = ""
    error: str = ""
    resume_retriable: bool = False
    stop_reason: str = ""
    points: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    result_summaries: list[dict[str, str]] = field(default_factory=list)
    task_description: str = ""
    finalists: list[dict[str, Any]] = field(default_factory=list)
    merge: dict[str, str] = field(default_factory=dict)
    artifact_preview: dict[str, Any] = field(default_factory=dict)
    related_works: list[dict[str, Any]] = field(default_factory=list)
    research_analysis: dict[str, Any] = field(default_factory=dict)
    report_agent: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
