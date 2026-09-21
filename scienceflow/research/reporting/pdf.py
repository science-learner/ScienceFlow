"""Direct PDF projection with no browser or external command dependency."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

from .lineage import lineage_layout
from .markdown import _duration, _time
from .models import ReportDocument
from .narrative import abstract, description_plain, findings, methods, outcome_label


def _metric_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Circle, Drawing, Line, String
    from reportlab.lib.colors import HexColor

    drawing = Drawing(500, 170)
    drawing.add(
        String(0, 154, "Metric history", fontSize=12, fillColor=HexColor("#333c4b"))
    )
    values = []
    for index, point in enumerate(report.points):
        try:
            x_value = float(point.get("x_value", index))
            metric = float(point.get("metric"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(metric):
            values.append((x_value, metric))
    if not values:
        drawing.add(
            String(
                18,
                82,
                "No comparable metric points recorded.",
                fillColor=HexColor("#737c8a"),
            )
        )
        return drawing
    drawing.add(Line(42, 28, 42, 138, strokeColor=HexColor("#536575")))
    drawing.add(Line(42, 28, 485, 28, strokeColor=HexColor("#536575")))
    x_min, x_max = min(x for x, _ in values), max(x for x, _ in values)
    y_min, y_max = min(y for _, y in values), max(y for _, y in values)
    x_span, y_span = max(1, x_max - x_min), max(1e-12, y_max - y_min)
    previous = None
    for x_value, metric in values:
        x = 42 + (x_value - x_min) / x_span * 443
        y = 28 + (metric - y_min) / y_span * 110
        if previous:
            drawing.add(
                Line(*previous, x, y, strokeColor=HexColor("#8fa9c4"), strokeWidth=1.8)
            )
        drawing.add(Circle(x, y, 2.3, fillColor=HexColor("#ef8398"), strokeColor=None))
        previous = (x, y)
    drawing.add(
        String(45, 31, f"{y_min:.5g}", fontSize=7, fillColor=HexColor("#737c8a"))
    )
    drawing.add(
        String(45, 128, f"{y_max:.5g}", fontSize=7, fillColor=HexColor("#737c8a"))
    )
    return drawing


def _event_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import HexColor

    values = [
        (key, value) for key, value in report.eec.items() if isinstance(value, int)
    ][:7]
    drawing = Drawing(500, max(90, 35 + len(values) * 20))
    drawing.add(
        String(
            0,
            drawing.height - 14,
            "Execution control events",
            fontSize=12,
            fillColor=HexColor("#333c4b"),
        )
    )
    maximum = max((value for _, value in values), default=1) or 1
    for index, (label, value) in enumerate(values):
        y = drawing.height - 38 - index * 20
        drawing.add(
            String(0, y + 3, label[:24], fontSize=7, fillColor=HexColor("#536575"))
        )
        drawing.add(
            Rect(
                125,
                y,
                value / maximum * 300,
                9,
                fillColor=HexColor("#b8a4e3"),
                strokeColor=None,
            )
        )
        drawing.add(
            String(432, y + 2, str(value), fontSize=7, fillColor=HexColor("#333c4b"))
        )
    if not values:
        drawing.add(
            String(18, 30, "No EEC events recorded.", fillColor=HexColor("#737c8a"))
        )
    return drawing


def _lineage_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib.colors import HexColor

    grouped = {}
    for stage in report.stages[:80]:
        worker = str(stage.get("worker") or stage.get("worker_id") or "worker")
        grouped.setdefault(worker, []).append(stage)
    rows = sorted(grouped.items())[:8]
    drawing_height = min(520, max(130, 42 + len(rows) * 225))
    drawing = Drawing(500, drawing_height)
    drawing.add(
        String(
            0,
            drawing.height - 14,
            "Stage lineage",
            fontSize=12,
            fillColor=HexColor("#333c4b"),
        )
    )
    if not rows:
        drawing.add(
            String(
                18,
                30,
                "No formal research stages recorded.",
                fillColor=HexColor("#737c8a"),
            )
        )
        return drawing
    available = drawing.height - 38
    panel_height = available / max(1, len(rows))
    for row_index, (worker, stages) in enumerate(rows):
        panel_top = drawing.height - 30 - row_index * panel_height
        by_id = {
            str(stage.get("id") or stage.get("candidate") or index): stage
            for index, stage in enumerate(stages)
        }
        local, local_width, lane_names = lineage_layout(by_id)
        max_lane = max(0, len(lane_names) - 1)
        lane_step = min(38, max(13, (panel_height - 34) / max(1, max_lane + 1)))
        x_scale = min(0.46, 455 / max(1000, local_width))
        positions = {
            identity: (
                20 + center * x_scale,
                panel_top - 34 - lane * lane_step,
            )
            for identity, (center, lane) in local.items()
        }
        drawing.add(
            Rect(
                4,
                panel_top - panel_height + 4,
                492,
                panel_height - 8,
                rx=6,
                ry=6,
                fillColor=HexColor("#f3f5f8"),
                strokeColor=HexColor("#d4d9e2"),
            )
        )
        drawing.add(
            String(
                9,
                panel_top - 15,
                worker[:12],
                fontSize=7.5,
                fillColor=HexColor("#6887a8"),
            )
        )
        for lane, lineage in enumerate(lane_names):
            drawing.add(
                String(
                    9,
                    panel_top - 37 - lane * lane_step,
                    lineage[:8],
                    fontSize=6.5,
                    fillColor=HexColor("#6887a8"),
                )
            )
        for identity, stage in by_id.items():
            x2, y2 = positions[identity]
            for source, restored in (
                (stage.get("parent"), False),
                (stage.get("restored"), True),
            ):
                if (
                    not source
                    or str(source) not in positions
                    or str(source) == identity
                ):
                    continue
                x1, y1 = positions[str(source)]
                if y1 == y2:
                    direction = 1 if x2 >= x1 else -1
                    start = (x1 + direction * 43, y1)
                    end = (x2 - direction * 43, y2)
                else:
                    direction = 1 if y2 > y1 else -1
                    start = (x1, y1 + direction * 10)
                    end = (x2, y2 - direction * 10)
                drawing.add(
                    Line(
                        *start,
                        *end,
                        strokeColor=HexColor("#c5667b" if restored else "#6887a8"),
                        strokeDashArray=[3, 2] if restored else None,
                    )
                )
        for identity, stage in by_id.items():
            x, y = positions[identity]
            label = str(stage.get("stage") or stage.get("lineage") or identity)[:8]
            phase = stage.get("kind") == "lineage_start"
            drawing.add(
                Rect(
                    x - 43,
                    y - 10,
                    86,
                    20,
                    rx=4,
                    ry=4,
                    fillColor=HexColor("#1b202b" if phase else "#202633"),
                    strokeColor=HexColor("#5d6575" if phase else "#455168"),
                    strokeDashArray=[4, 3] if phase else None,
                )
            )
            drawing.add(
                Rect(
                    x - 43,
                    y - 6,
                    2,
                    12,
                    fillColor=HexColor("#b8a4e3"),
                    strokeColor=None,
                )
            )
            drawing.add(
                String(
                    x - 36,
                    y - 3,
                    label,
                    fontSize=7,
                    fillColor=HexColor("#eee7dc"),
                )
            )
    return drawing


def _related_work_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Circle, Drawing, Line, String
    from reportlab.lib.colors import HexColor

    values = [
        item
        for item in report.related_works
        if isinstance(item.get("year"), int) and isinstance(item.get("citations"), int)
    ]
    drawing = Drawing(500, 175)
    drawing.add(
        String(
            0,
            159,
            "Related research landscape",
            fontSize=12,
            fillColor=HexColor("#333c4b"),
        )
    )
    if not values:
        drawing.add(
            String(
                18,
                82,
                "No related-work metadata retrieved.",
                fillColor=HexColor("#737c8a"),
            )
        )
        return drawing
    drawing.add(Line(42, 28, 42, 138, strokeColor=HexColor("#536575")))
    drawing.add(Line(42, 28, 485, 28, strokeColor=HexColor("#536575")))
    years = [int(item["year"]) for item in values]
    weights = [math.log1p(int(item["citations"])) for item in values]
    year_span = max(1, max(years) - min(years))
    weight_span = max(1.0, max(weights) - min(weights))
    colors = ("#6887a8", "#816cae", "#a47d45", "#6f9486")
    for index, item in enumerate(values):
        x = 52 + (int(item["year"]) - min(years)) / year_span * 420
        y = 38 + (math.log1p(int(item["citations"])) - min(weights)) / weight_span * 88
        radius = 4 + min(8, math.sqrt(max(0, int(item["citations"]))) * 0.3)
        drawing.add(
            Circle(
                x,
                y,
                radius,
                fillColor=HexColor(colors[index % len(colors)]),
                strokeColor=None,
            )
        )
        drawing.add(
            String(
                x - 5,
                y - 2,
                str(item.get("id") or "R?"),
                fontSize=6,
                fillColor=HexColor("#ffffff"),
            )
        )
    return drawing


def _candidate_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import HexColor

    values = [
        value
        for value in report.finalists
        if isinstance(value.get("metric"), (int, float))
    ]
    drawing = Drawing(500, 175)
    drawing.add(
        String(
            0,
            159,
            "Final candidate comparison",
            fontSize=12,
            fillColor=HexColor("#333c4b"),
        )
    )
    if not values:
        drawing.add(
            String(
                18,
                82,
                "No evaluated finalists recorded.",
                fillColor=HexColor("#737c8a"),
            )
        )
        return drawing
    metrics = [float(value["metric"]) for value in values]
    low, high = min(metrics), max(metrics)
    span = max(high - low, max(abs(low), abs(high), 1) * 0.02)
    baseline = low - span * 0.08
    colors = ("#816cae", "#6887a8", "#a47d45", "#6f9486")
    slot = 430 / len(values)
    for index, value in enumerate(values):
        metric = float(value["metric"])
        height = max(7, (metric - baseline) / max(high - baseline, 1e-12) * 105)
        x = 42 + index * slot + slot * 0.22
        width = min(55, slot * 0.56)
        drawing.add(
            Rect(
                x,
                28,
                width,
                height,
                rx=3,
                ry=3,
                fillColor=HexColor(colors[index % len(colors)]),
                strokeColor=None,
            )
        )
        drawing.add(
            String(
                x,
                16,
                str(value.get("candidate_id") or "candidate")[:12],
                fontSize=7,
                fillColor=HexColor("#536575"),
            )
        )
        drawing.add(
            String(
                x,
                min(145, 34 + height),
                f"{metric:.6g}",
                fontSize=7,
                fillColor=HexColor("#333c4b"),
            )
        )
    return drawing


def _artifact_drawing(report: ReportDocument):
    from reportlab.graphics.shapes import Circle, Drawing, Rect, String
    from reportlab.lib.colors import HexColor

    if report.artifact_preview.get("kind") != "circle_packing":
        return None
    circles = report.artifact_preview.get("circles") or []
    drawing = Drawing(500, 300)
    drawing.add(
        String(
            0,
            284,
            "Best result artifact",
            fontSize=12,
            fillColor=HexColor("#333c4b"),
        )
    )
    left, bottom, size = 120, 20, 250
    drawing.add(
        Rect(
            left,
            bottom,
            size,
            size,
            fillColor=HexColor("#f4f5f8"),
            strokeColor=HexColor("#6887a8"),
        )
    )
    colors = ("#6887a8", "#816cae", "#a47d45", "#6f9486")
    for index, circle in enumerate(circles):
        x, y, radius = (float(value) for value in circle)
        drawing.add(
            Circle(
                left + x * size,
                bottom + y * size,
                max(0, radius * size),
                fillColor=HexColor(colors[index % len(colors)] + "55"),
                strokeColor=HexColor(colors[index % len(colors)]),
                strokeWidth=0.7,
            )
        )
    return drawing


def render_pdf(report: ReportDocument, target: str | Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        body_font = "STSong-Light"
    except (KeyError, TypeError, ValueError):
        body_font = "Helvetica"
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ScienceFlowBody",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=9.5,
        leading=14,
    )
    heading = ParagraphStyle(
        "ScienceFlowHeading",
        parent=styles["Heading2"],
        fontName=body_font,
        textColor=colors.HexColor("#333c4b"),
        spaceBefore=12,
    )
    title = ParagraphStyle(
        "ScienceFlowTitle",
        parent=styles["Title"],
        fontName=body_font,
        textColor=colors.HexColor("#816cae"),
        alignment=TA_CENTER,
    )
    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    story = [
        Paragraph("ScienceFlow Scientific Research Report", title),
        Paragraph(escape(report.task_name), heading),
        Paragraph(
            f"Attempt {report.attempt} · {escape(outcome_label(report))} · Generated {escape(report.generated_at_utc)}",
            body,
        ),
        Spacer(1, 12),
        Paragraph("Abstract", heading),
        Paragraph(escape(abstract(report)), body),
        Paragraph("Research question and objective", heading),
    ]
    for line in description_plain(report.task_description) or [
        "No task description was preserved with this run."
    ]:
        story.append(Paragraph(escape(line), body))
    if report.related_works:
        story.append(Paragraph("Related research", heading))
        story.append(
            Paragraph(
                escape(
                    str(
                        report.research_analysis.get("related_work")
                        or "Retrieved publications are potentially related context and require relevance verification."
                    )
                ),
                body,
            )
        )
        story.append(_related_work_drawing(report))
        for item in report.related_works:
            story.append(
                Paragraph(
                    escape(
                        f"[{item.get('id', 'R?')}] {item.get('title', 'Untitled')} "
                        f"({item.get('year') or 'n.d.'}); DOI {item.get('doi') or '—'}"
                    ),
                    body,
                )
            )
    story.append(Paragraph("Experimental design", heading))
    for value in methods(report):
        story.append(Paragraph("• " + escape(value), body))
    rows = [
        ["Outcome", outcome_label(report)],
        ["Run ID", report.run_id],
        ["Started", _time(report.started_at)],
        ["Finished", _time(report.finished_at)],
        [
            "Elapsed / budget",
            f"{_duration(report.elapsed_sec)} / {_duration(report.budget_sec)}",
        ],
        [report.metric_name + (" ↓" if report.lower_is_better else " ↑"), report.best],
        ["Evaluated / valid", f"{report.evaluated} / {report.valid}"],
        ["Cost", report.cost],
        ["GPU", report.gpu],
    ]
    table = Table(
        [
            [Paragraph(escape(str(a)), body), Paragraph(escape(str(b)), body)]
            for a, b in rows
        ],
        colWidths=(45 * mm, 115 * mm),
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d4d8df")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef1f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 12),
            Paragraph("Results", heading),
            *[Paragraph("• " + escape(value), body) for value in findings(report)],
            _metric_drawing(report),
            _candidate_drawing(report),
        ]
    )
    artifact = _artifact_drawing(report)
    if artifact is not None:
        story.append(artifact)
    story.extend(
        [
            Paragraph("Research trajectory", heading),
            _lineage_drawing(report),
        ]
    )
    story.append(
        Paragraph("ESTRA: " + escape(str(report.estra or "No events recorded")), body)
    )
    story.extend(
        [
            Paragraph("Execution and resource evidence", heading),
            _event_drawing(report),
        ]
    )
    story.append(Paragraph(escape(report.usage or "No provider usage reported"), body))
    story.append(
        Paragraph(
            f"EEC events: {report.eec_total} · {escape(str(report.eec or 'No events recorded'))}",
            body,
        )
    )
    story.append(Paragraph("Worker outcomes", heading))
    for worker in report.workers or ["No worker progress records were available."]:
        story.append(Paragraph("• " + escape(worker), body))
    if report.failure_kind or report.error or report.stop_reason:
        story.append(Paragraph("Failure and recovery evidence", heading))
        story.append(
            Paragraph(
                escape(
                    f"Failure: {report.failure_kind or '—'} · Stop: {report.stop_reason or '—'}"
                ),
                body,
            )
        )
        story.append(Paragraph(escape(report.error or "No error text recorded"), body))
    if report.result_summaries:
        story.extend([PageBreak(), Paragraph("Research notes", heading)])
        for item in report.result_summaries:
            story.append(Paragraph(escape(item["worker"]), heading))
            story.append(Paragraph(escape(item["text"]).replace("\n", "<br/>"), body))
    story.append(Paragraph("Interpretation and limitations", heading))
    story.append(
        Paragraph(
            escape(
                str(
                    report.research_analysis.get("interpretation")
                    or report.research_analysis.get("summary")
                    or "Quantitative claims are restricted to persisted evaluator records. Candidate-generation notes describe search decisions; only evaluator-accepted candidates are treated as results."
                )
            ),
            body,
        )
    )
    for value in report.research_analysis.get("limitations") or []:
        story.append(Paragraph("• " + escape(str(value)), body))
    story.append(Paragraph("Reproducibility and provenance", heading))
    story.append(
        Paragraph(
            "Generated from persisted ScienceFlow evidence. Report agent: "
            + escape(str(report.report_agent.get("status") or "not used"))
            + ". Related publications are retrieval candidates and do not replace evaluator evidence.",
            body,
        )
    )
    document.build(story)
