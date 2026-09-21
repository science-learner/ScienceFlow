"""Markdown projection for a final research report."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from .models import ReportDocument


def _time(value: float | None) -> str:
    if value is None:
        return "—"
    return datetime.fromtimestamp(value, UTC).isoformat()


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}h {minutes:02d}m {seconds:02d}s"
        if hours
        else f"{minutes}m {seconds:02d}s"
    )


def _table(rows: list[tuple[str, object]]) -> str:
    lines = []
    for key, value in rows:
        escaped = str(value).replace("|", "\\|")
        lines.append(f"| {key} | {escaped} |")
    return "\n".join(lines)


def _code_fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _nested_description(text: str) -> str:
    """Keep task-description headings below the report section heading."""
    return re.sub(
        r"(?m)^(#{1,6})(\s+)",
        lambda match: "#" * min(6, len(match[1]) + 2) + match[2],
        text,
    )


def render_markdown(report: ReportDocument) -> str:
    from .narrative import abstract, findings, methods, outcome_label

    arrow = "↓" if report.lower_is_better else "↑"
    related_offset = 1 if report.related_works else 0
    metric_figure = 1 + related_offset
    candidate_figure = metric_figure + 1
    artifact_figure = candidate_figure + 1
    trajectory_figure = (
        artifact_figure + 1 if report.artifact_preview else candidate_figure + 1
    )
    resource_figure = trajectory_figure + 1
    rows = [
        ("Outcome", outcome_label(report)),
        ("Run ID", f"`{report.run_id}`"),
        ("Attempt", report.attempt),
        ("Started", _time(report.started_at)),
        ("Finished", _time(report.finished_at)),
        (
            "Elapsed / budget",
            f"{_duration(report.elapsed_sec)} / {_duration(report.budget_sec)}",
        ),
    ]
    lines = [
        f"# {report.task_name}",
        "",
        "**ScienceFlow Scientific Research Report**  ",
        f"Generated {_time(report.finished_at or report.started_at)} · Attempt {report.attempt}",
        "",
        "## Abstract",
        "",
        abstract(report),
        "",
        "## Research question and objective",
        "",
        _nested_description(report.task_description)
        if report.task_description
        else "No task description was preserved with this run.",
    ]
    if report.related_works:
        lines.extend(
            [
                "",
                "## Related research",
                "",
                report.research_analysis.get("related_work")
                or "The following publications were retrieved as potentially related context; relevance should be verified before citation.",
                "",
                "![Related research landscape](report_assets/related-work.svg)",
                "",
                "*Figure 1. Retrieved publications by year and citation count; marker area scales with citation count.*",
                "",
            ]
        )
        for item in report.related_works:
            authors = ", ".join(item.get("authors") or []) or "Unknown authors"
            title = (
                str(item.get("title") or "Untitled")
                .replace("[", "\\[")
                .replace("]", "\\]")
            )
            link = item.get("url") or ""
            title = f"[{title}]({link})" if link else title
            lines.append(
                f"- **{item.get('id', 'R?')}** {authors} ({item.get('year') or 'n.d.'}). {title}. {item.get('venue') or ''}"
            )
    lines.extend(
        [
            "",
            "## Experimental design",
            "",
            *[f"- {value}" for value in methods(report)],
            "",
            "### Run protocol",
            "",
            "| Field | Value |",
            "| --- | --- |",
            _table(rows),
            "",
            "## Results",
            "",
            *[f"- {value}" for value in findings(report)],
            "",
            "![Metric history](report_assets/metric-history.svg)",
            "",
            f"*Figure {metric_figure}. Evaluator-backed {report.metric_name} trajectory over elapsed research time.*",
            "",
            "### Final candidate comparison",
            "",
            "| Rank | Candidate | Source stage | Metric | Valid | Evaluator | Artifact SHA |",
            "| ---: | --- | --- | ---: | :---: | --- | --- |",
        ]
    )
    for finalist in report.finalists:
        metric = finalist.get("metric")
        metric_text = "—" if metric is None else f"{metric:.8g}"
        lines.append(
            f"| {finalist.get('rank', '—')} | {finalist.get('candidate_id', '—')} | "
            f"{finalist.get('source') or '—'} | {metric_text} | "
            f"{'yes' if finalist.get('valid') else 'no'} | "
            f"{finalist.get('evaluator') or '—'} | `{finalist.get('artifact_sha') or '—'}` |"
        )
    if not report.finalists:
        lines.append("| — | No finalists recorded | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "![Final candidate comparison](report_assets/candidate-comparison.svg)",
            "",
            f"*Figure {candidate_figure}. Final candidate comparison; {arrow} indicates the preferred metric direction.*",
        ]
    )
    if report.artifact_preview:
        lines.extend(
            [
                "",
                "### Best result artifact",
                "",
                "![Best result artifact](report_assets/result-artifact.svg)",
                "",
                f"*Figure {artifact_figure}. Structured visualization of the highest-ranked validated final artifact.*",
            ]
        )
    lines.extend(
        [
            "",
            "## Research trajectory",
            "",
            "![Stage lineage](report_assets/stage-lineage.svg)",
            "",
            f"*Figure {trajectory_figure}. Recorded stages grouped by worker into horizontal lineage lanes; dashed edges denote restoration.*",
            "",
            "| ESTRA signal | Count |",
            "| --- | ---: |",
            *[f"| {key} | {value} |" for key, value in sorted(report.estra.items())],
            "",
            "### Worker outcomes",
            "",
            *(
                [f"- {worker}" for worker in report.workers]
                or ["- No worker progress records were available."]
            ),
            "",
            "## Execution and resource evidence",
            "",
            "![Execution control events](report_assets/resource-timeline.svg)",
            "",
            f"*Figure {resource_figure}. Evidence-aware execution-control events recorded during the run.*",
            "",
            f"- Usage: {report.usage or 'No provider usage reported'}",
            f"- EEC events: {report.eec_total}",
            f"- GPU allocation: {report.gpu}",
            f"- Estimated cost: {report.cost}",
            "",
            "## Interpretation and limitations",
            "",
            report.research_analysis.get("interpretation")
            or report.research_analysis.get("summary")
            or (
                "The quantitative claims above are restricted to persisted evaluator "
                "records. Candidate generation notes describe search decisions, while "
                "only candidates accepted by the configured evaluator are treated as "
                "results."
            ),
        ]
    )
    limitations = report.research_analysis.get("limitations") or []
    if limitations:
        lines.extend(["", "### Report-agent limitations", ""])
        lines.extend(f"- {value}" for value in limitations)
    if report.failure_kind or report.error or report.stop_reason:
        lines.extend(
            [
                "",
                "### Termination and recovery",
                "",
                f"- Failure kind: {report.failure_kind or '—'}",
                f"- Error: {report.error or '—'}",
                f"- Stop reason: {report.stop_reason or '—'}",
                f"- Resume retriable: {'yes' if report.resume_retriable else 'no'}",
            ]
        )
    if report.result_summaries:
        lines.extend(["", "## Research notes"])
        for summary in report.result_summaries:
            lines.extend(
                ["", f"### {summary['worker']}", "", _code_fence(summary["text"])]
            )
    if report.recent:
        lines.extend(
            [
                "",
                "## Control evidence appendix",
                "",
                _code_fence("\n".join(report.recent)),
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility and provenance",
            "",
            f"- Report schema: {report.schema_version}",
            f"- Generated: {report.generated_at_utc}",
            f"- Metric: `{report.metric_name}` ({'lower' if report.lower_is_better else 'higher'} is better)",
            f"- Merge mode: {report.merge.get('merge_mode', '—')}",
            f"- Valid finalists: {report.merge.get('valid_final_count', '—')} / {report.merge.get('final_count', '—')}",
            "- Evidence source: persisted task description, evaluator outputs, stage lineage, final reduction, ESTRA, EEC and provider-usage records.",
            f"- Report agent: {report.report_agent.get('status', 'not used')}"
            + (
                f" · model `{report.report_agent.get('model')}`"
                if report.report_agent.get("model")
                else ""
            ),
            "- Related publications are retrieval candidates; experimental claims remain tied to persisted evaluator evidence.",
            "",
        ]
    )
    return "\n".join(lines)
