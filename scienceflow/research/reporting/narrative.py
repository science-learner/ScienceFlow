"""Evidence-derived prose shared by the report renderers."""

from __future__ import annotations

import math
import re

from .markdown import _duration
from .models import ReportDocument


def outcome_label(report: ReportDocument) -> str:
    if report.status == "completed" and report.valid:
        return "Completed with a valid result"
    if report.valid:
        return f"{report.status.capitalize()} with recoverable valid results"
    return report.status.capitalize()


def _metrics(report: ReportDocument) -> list[tuple[float, float, str, str]]:
    values = []
    for index, point in enumerate(report.points):
        if point.get("valid") is False:
            continue
        try:
            metric = float(point.get("metric"))
            position = float(point.get("x_value", index))
        except (TypeError, ValueError):
            continue
        if math.isfinite(metric) and math.isfinite(position):
            values.append(
                (
                    position,
                    metric,
                    str(point.get("worker_id") or "worker"),
                    str(point.get("stage_id") or "stage"),
                )
            )
    return sorted(values)


def abstract(report: ReportDocument) -> str:
    direction = "minimize" if report.lower_is_better else "maximize"
    stage_count = len(
        {
            (
                stage.get("worker") or stage.get("worker_id"),
                stage.get("stage") or stage.get("stage_id"),
            )
            for stage in report.stages
            if stage.get("kind") != "lineage_start"
        }
    )
    return (
        f"This study used ScienceFlow to {direction} {report.metric_name} for "
        f"{report.task_name}. {len(report.workers)} parallel workers explored "
        f"{stage_count} evaluator-backed stages over {_duration(report.elapsed_sec)}. "
        f"The run evaluated {report.evaluated} candidates, accepted {report.valid} as "
        f"valid, and obtained a best recorded value of {report.best}. "
        f"The run outcome was {outcome_label(report).lower()}."
    )


def methods(report: ReportDocument) -> list[str]:
    direction = "lower" if report.lower_is_better else "higher"
    values = [
        f"Parallel long-horizon search used {len(report.workers)} workers with a budget of {_duration(report.budget_sec)}.",
        f"Candidate quality was determined by the configured evaluator using {report.metric_name}; {direction} values are preferred.",
    ]
    decisions = report.estra.get("estra_decision", 0)
    switches = report.estra.get("estra_stage_switched", 0)
    compactions = report.estra.get("estra_keep_current_compacted", 0)
    values.append(
        f"Adaptive research control recorded {decisions} ESTRA decisions, {switches} lineage switches, and {compactions} context compactions."
    )
    if report.finalists:
        valid = sum(bool(value.get("valid")) for value in report.finalists)
        mode = report.merge.get("merge_mode", "the configured reduction policy")
        values.append(
            f"Final reduction used {mode} to compare {len(report.finalists)} finalists; {valid} passed final validation."
        )
    return values


def findings(report: ReportDocument) -> list[str]:
    values = []
    metrics = _metrics(report)
    if metrics:
        first = metrics[0][1]
        best = (
            min(metrics, key=lambda item: item[1])
            if report.lower_is_better
            else max(metrics, key=lambda item: item[1])
        )
        delta = best[1] - first
        improvement = -delta if report.lower_is_better else delta
        values.append(
            f"The best evaluator-backed stage was {best[2]}:{best[3]} at {best[1]:.6g}; improvement over the first valid observation was {improvement:+.6g}."
        )
    rate = report.valid / report.evaluated if report.evaluated else None
    if rate is not None:
        values.append(
            f"{report.valid} of {report.evaluated} evaluated candidates were valid ({rate:.1%})."
        )
    if report.finalists:
        best = report.finalists[0]
        values.append(
            f"Final candidate {best['candidate_id']} ranked first with {report.metric_name}={best['metric']:.6g} and evaluator status {best['evaluator']}."
        )
    if report.stop_reason:
        values.append(f"The recorded termination reason was {report.stop_reason}.")
    return values or ["No evaluator-backed quantitative finding was available."]


def description_plain(text: str) -> list[str]:
    values = []
    in_code = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line or in_code:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*]\s+", "• ", line)
        line = line.replace("`", "")
        values.append(line)
    return values
