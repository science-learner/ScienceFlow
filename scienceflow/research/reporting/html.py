"""Self-contained scientific HTML report with escaped persisted evidence."""

from __future__ import annotations

import re
from html import escape

from .markdown import _duration, _time
from .models import ReportDocument
from .narrative import abstract, findings, methods, outcome_label


def _stat(label: str, value: object) -> str:
    return (
        '<div class="stat"><small>'
        + escape(label)
        + "</small><strong>"
        + escape(str(value))
        + "</strong></div>"
    )


def _bullets(values: list[str], empty: str = "No evidence was recorded.") -> str:
    if not values:
        return f'<p class="muted">{escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in values) + "</ul>"


def _description_html(value: str) -> str:
    if not value:
        return '<p class="muted">No task description was preserved with this run.</p>'
    blocks: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] = []
    in_code = False

    def flush() -> None:
        if paragraph:
            blocks.append("<p>" + escape(" ".join(paragraph)) + "</p>")
            paragraph.clear()
        if bullets:
            blocks.append(
                "<ul>"
                + "".join(f"<li>{escape(item)}</li>" for item in bullets)
                + "</ul>"
            )
            bullets.clear()

    for raw in value.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            if in_code:
                blocks.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
                code.clear()
            else:
                flush()
            in_code = not in_code
            continue
        if in_code:
            code.append(raw)
            continue
        if not line:
            flush()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush()
            level = min(4, len(heading.group(1)) + 2)
            blocks.append(f"<h{level}>{escape(heading.group(2))}</h{level}>")
        elif re.match(r"^[-*]\s+", line):
            if paragraph:
                flush()
            bullets.append(re.sub(r"^[-*]\s+", "", line))
        else:
            if bullets:
                flush()
            paragraph.append(line.replace("`", ""))
    flush()
    if code:
        blocks.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
    return "".join(blocks)


def _finalist_table(report: ReportDocument) -> str:
    rows = []
    for finalist in report.finalists:
        metric = finalist.get("metric")
        metric_text = "—" if metric is None else f"{metric:.8g}"
        rows.append(
            "<tr>"
            f"<td>{finalist.get('rank', '—')}</td>"
            f"<td>{escape(str(finalist.get('candidate_id') or '—'))}</td>"
            f"<td>{escape(str(finalist.get('source') or '—'))}</td>"
            f"<td>{escape(metric_text)}</td>"
            f"<td>{'yes' if finalist.get('valid') else 'no'}</td>"
            f"<td>{escape(str(finalist.get('evaluator') or '—'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(
            '<tr><td colspan="6" class="muted">No finalists were recorded.</td></tr>'
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Rank</th>'
        "<th>Candidate</th><th>Source stage</th><th>Metric</th><th>Valid</th>"
        "<th>Evaluator</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _figure(svg: str, caption: str) -> str:
    return f"<figure>{svg}<figcaption>{escape(caption)}</figcaption></figure>"


def _related_work(report: ReportDocument, svg: str) -> str:
    if not report.related_works:
        return ""
    rows = []
    for item in report.related_works:
        title = escape(str(item.get("title") or "Untitled"))
        url = str(item.get("url") or "")
        if url.startswith("https://doi.org/"):
            title = f'<a href="{escape(url, quote=True)}" target="_blank" rel="noreferrer">{title}</a>'
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('id') or 'R?'))}</td>"
            f"<td>{title}</td>"
            f"<td>{escape(str(item.get('year') or '—'))}</td>"
            f"<td>{escape(str(item.get('venue') or '—'))}</td>"
            f"<td>{escape(str(item.get('citations') or 0))}</td>"
            "</tr>"
        )
    analysis = report.research_analysis.get("related_work") or (
        "These publications were retrieved as potentially related context; "
        "their relevance should be verified before citation."
    )
    return (
        "<section><h2>Related research</h2>"
        f"<p>{escape(str(analysis))}</p>"
        + _figure(
            svg,
            "Retrieved publications by year and citation count; marker area scales with citation count.",
        )
        + '<div class="table-scroll"><table><thead><tr><th>ID</th><th>Publication</th>'
        "<th>Year</th><th>Venue</th><th>Citations</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def render_html(report: ReportDocument, svgs: dict[str, str]) -> str:
    arrow = "↓" if report.lower_is_better else "↑"
    related_offset = 1 if report.related_works else 0
    metric_figure = 1 + related_offset
    candidate_figure = metric_figure + 1
    artifact_figure = candidate_figure + 1
    trajectory_figure = (
        artifact_figure + 1 if report.artifact_preview else candidate_figure + 1
    )
    resource_figure = trajectory_figure + 1
    artifact = ""
    if report.artifact_preview:
        artifact = "<h3>Best result artifact</h3>" + _figure(
            svgs["result-artifact.svg"],
            f"Figure {artifact_figure}. Structured view of the highest-ranked validated final artifact.",
        )
    failure = ""
    if report.failure_kind or report.error or report.stop_reason:
        failure = (
            '<aside class="evidence"><h3>Termination and recovery</h3>'
            + _bullets(
                [
                    f"Failure kind: {report.failure_kind or '—'}",
                    f"Stop reason: {report.stop_reason or '—'}",
                    f"Resume retriable: {'yes' if report.resume_retriable else 'no'}",
                    f"Recorded error: {report.error or '—'}",
                ]
            )
            + "</aside>"
        )
    notes = ""
    if report.result_summaries:
        notes = (
            "<section><h2>Research notes</h2>"
            + "".join(
                f"<details><summary>{escape(item['worker'])}</summary>"
                f"<pre>{escape(item['text'])}</pre></details>"
                for item in report.result_summaries
            )
            + "</section>"
        )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{escape(report.task_name)} · ScienceFlow scientific report</title>
<style>
:root{{--paper:#fbfaf7;--ink:#242632;--muted:#68707d;--line:#d8dbe2;--soft:#f0f1f5;--blue:#6887a8;--purple:#816cae;--gold:#a47d45}}
*{{box-sizing:border-box}}body{{margin:0;background:#e8e9ee;color:var(--ink);font:15px/1.72 system-ui,-apple-system,sans-serif}}
article{{max-width:1040px;margin:32px auto;background:var(--paper);box-shadow:0 12px 38px #30344120;padding:64px 76px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:26px;margin-bottom:38px}}.kicker{{color:var(--purple);font-size:12px;font-weight:700;letter-spacing:.13em;text-transform:uppercase}}
h1{{font:700 38px/1.18 Georgia,serif;margin:12px 0}}.meta,.muted,small,figcaption{{color:var(--muted)}}h2{{font:700 23px/1.3 Georgia,serif;margin:42px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px}}h3{{font:700 17px/1.4 Georgia,serif;margin:26px 0 10px}}
.abstract{{font:17px/1.75 Georgia,serif;border-left:3px solid var(--purple);padding:2px 0 2px 20px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:18px 0}}.stat{{background:var(--paper);padding:13px 15px}}.stat small{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}.stat strong{{display:block;margin-top:3px;overflow-wrap:anywhere}}
figure{{margin:24px 0 30px}}figure svg{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}figcaption{{font:13px/1.5 Georgia,serif;margin-top:8px;text-align:center}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--soft);font-weight:650}}.table-scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);border:1px solid var(--line);padding:14px;border-radius:7px;max-height:420px;overflow:auto}}code{{font:13px ui-monospace,monospace}}li{{margin:5px 0}}.evidence{{background:#f4f0e9;border-left:3px solid var(--gold);padding:12px 18px;margin:24px 0}}details{{border-top:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:650}}
.provenance{{font-size:13px;color:var(--muted)}}
@media(max-width:720px){{article{{margin:0;padding:32px 20px}}h1{{font-size:30px}}.stats{{grid-template-columns:1fr 1fr}}}}
@media print{{body{{background:white}}article{{margin:0;box-shadow:none;padding:12mm}}section,figure,.stats{{break-inside:avoid}}}}
</style><article>
<header><div class="kicker">ScienceFlow · Scientific Research Report</div><h1>{escape(report.task_name)}</h1><p class="meta">Attempt {report.attempt} · {escape(outcome_label(report))} · generated {escape(report.generated_at_utc)}</p></header>
<section><h2>Abstract</h2><p class="abstract">{escape(abstract(report))}</p></section>
<section><h2>Research question and objective</h2>{_description_html(report.task_description)}</section>
{_related_work(report, svgs["related-work.svg"])}
<section><h2>Experimental design</h2>{_bullets(methods(report))}<div class="stats">
{_stat("Outcome", outcome_label(report))}{_stat("Elapsed / budget", _duration(report.elapsed_sec) + " / " + _duration(report.budget_sec))}{_stat("Workers", len(report.workers))}{_stat("Started", _time(report.started_at))}{_stat("Finished", _time(report.finished_at))}{_stat("Estimated cost", report.cost)}
</div></section>
<section><h2>Results</h2>{_bullets(findings(report))}
{_figure(svgs["metric-history.svg"], f"Figure {metric_figure}. Evaluator-backed {report.metric_name} trajectory over elapsed research time.")}
<h3>Final candidate comparison</h3>{_finalist_table(report)}
{_figure(svgs["candidate-comparison.svg"], f"Figure {candidate_figure}. Validated finalists; {arrow} indicates the preferred direction.")}
{artifact}</section>
<section><h2>Research trajectory</h2>
{_figure(svgs["stage-lineage.svg"], f"Figure {trajectory_figure}. Recorded stages grouped into horizontal lineage lanes; dashed edges denote restoration.")}
<h3>Worker outcomes</h3>{_bullets(report.workers, "No worker progress records were available.")}</section>
<section><h2>Execution and resource evidence</h2>
{_figure(svgs["resource-timeline.svg"], f"Figure {resource_figure}. Evidence-aware execution-control events recorded during the run.")}
{_bullets([report.usage or "No provider usage reported", f"GPU allocation: {report.gpu}", f"EEC events: {report.eec_total}"])}</section>
<section><h2>Interpretation and limitations</h2><p>{escape(str(report.research_analysis.get("interpretation") or report.research_analysis.get("summary") or "The quantitative claims are restricted to persisted evaluator records. Candidate-generation notes describe search decisions; only evaluator-accepted candidates are treated as scientific results."))}</p>{_bullets(report.research_analysis.get("limitations") or [], "")}{failure}</section>
{notes}
<section class="provenance"><h2>Reproducibility and provenance</h2><p>Run <code>{escape(report.run_id)}</code> · metric <code>{escape(report.metric_name)}</code> ({"lower" if report.lower_is_better else "higher"} is better) · merge mode {escape(report.merge.get("merge_mode", "—"))}.</p><p>Generated from persisted evaluator evidence. Report agent: {escape(report.report_agent.get("status", "not used"))}{" · " + escape(report.report_agent.get("model", "")) if report.report_agent.get("model") else ""}. Related publications are retrieval candidates and do not replace evaluator evidence.</p></section>
</article></html>"""
