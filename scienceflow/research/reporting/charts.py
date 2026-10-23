"""Small dependency-free SVG figures for portable final reports."""

from __future__ import annotations

import math
from collections import defaultdict
from html import escape

from .lineage import lineage_layout
from .models import ReportDocument

_COLORS = ("#8fa9c4", "#b8a4e3", "#ef8398", "#d8be91", "#87b6a7")


def _frame(title: str, body: str, *, height: int = 280, width: int = 800) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img">'
        f'<rect width="{width}" height="100%" rx="12" fill="#191e2a"/>'
        f'<text x="28" y="34" fill="#eee7dc" font-size="17" font-family="sans-serif">{escape(title)}</text>'
        f"{body}</svg>"
    )


def _empty(title: str, message: str) -> str:
    return _frame(
        title,
        f'<text x="28" y="112" fill="#9ca8b8" font-size="14" font-family="sans-serif">{escape(message)}</text>',
        height=160,
    )


def metric_svg(report: ReportDocument) -> str:
    values = []
    for index, point in enumerate(report.points):
        try:
            metric = float(point.get("metric"))
            x_value = float(point.get("x_value", index))
        except (TypeError, ValueError):
            continue
        if math.isfinite(metric) and math.isfinite(x_value):
            values.append((x_value, metric, str(point.get("worker_id") or "worker")))
    if not values:
        return _empty("Metric history", "No comparable metric points were recorded.")
    x_min, x_max = min(v[0] for v in values), max(v[0] for v in values)
    y_min, y_max = min(v[1] for v in values), max(v[1] for v in values)
    x_span, y_span = max(x_max - x_min, 1), max(y_max - y_min, 1e-12)
    grouped = defaultdict(list)
    for x_value, metric, worker in values:
        x = 70 + (x_value - x_min) / x_span * 690
        y = 225 - (metric - y_min) / y_span * 155
        grouped[worker].append((x, y))
    body = (
        '<line x1="70" y1="70" x2="70" y2="225" stroke="#536575"/>'
        '<line x1="70" y1="225" x2="760" y2="225" stroke="#536575"/>'
        f'<text x="70" y="250" fill="#9ca8b8" font-size="12">time</text>'
        f'<text x="74" y="88" fill="#9ca8b8" font-size="12">{escape(report.metric_name)}</text>'
    )
    for index, (worker, points) in enumerate(sorted(grouped.items())):
        color = _COLORS[index % len(_COLORS)]
        points.sort()
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body += f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        body += "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
            for x, y in points
        )
        body += f'<text x="{610 + index % 3 * 58}" y="54" fill="{color}" font-size="11">{escape(worker)}</text>'
    body += f'<text x="74" y="218" fill="#9ca8b8" font-size="11">{y_min:.5g}</text>'
    body += f'<text x="74" y="103" fill="#9ca8b8" font-size="11">{y_max:.5g}</text>'
    return _frame("Metric history", body)


def lineage_svg(report: ReportDocument) -> str:
    grouped = defaultdict(list)
    for stage in report.stages[:120]:
        grouped[str(stage.get("worker") or stage.get("worker_id") or "worker")].append(
            stage
        )
    if not grouped:
        return _empty("Stage lineage", "No formal research stages were recorded.")
    layouts = []
    top = 60
    width = 1000
    for worker, stages in sorted(grouped.items()):
        by_id = {
            str(stage.get("id") or stage.get("candidate") or index): stage
            for index, stage in enumerate(stages)
        }
        local, local_width, lane_names = lineage_layout(by_id)
        positions = {
            identity: (center, top + 78 + lane * 112)
            for identity, (center, lane) in local.items()
        }
        panel_height = 44 + len(lane_names) * 112
        layouts.append((worker, by_id, positions, top, panel_height, lane_names))
        width = max(width, int(local_width))
        top += panel_height + 16
    height = max(180, top + 12)
    body = (
        "<defs>"
        '<marker id="report-parent-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#8fa9c4"/></marker>'
        '<marker id="report-restore-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#ef8398"/></marker>'
        "</defs>"
        '<line x1="600" y1="29" x2="624" y2="29" stroke="#8fa9c4" marker-end="url(#report-parent-arrow)"/>'
        '<text x="632" y="33" fill="#9ca8b8" font-size="11">Parent → child</text>'
        '<line x1="765" y1="29" x2="789" y2="29" stroke="#ef8398" stroke-dasharray="5 5" marker-end="url(#report-restore-arrow)"/>'
        '<text x="797" y="33" fill="#9ca8b8" font-size="11">Restored from</text>'
    )
    for worker, by_id, positions, panel_top, panel_height, lane_names in layouts:
        body += f'<rect x="12" y="{panel_top}" width="{width - 24}" height="{panel_height}" rx="12" fill="#111620" stroke="#293140"/>'
        body += f'<text x="30" y="{panel_top + 23}" fill="#8fa9c4" font-size="13" font-weight="700">{escape(worker)}</text>'
        for lane, lineage in enumerate(lane_names):
            y = panel_top + 34 + lane * 112
            body += f'<rect x="22" y="{y}" width="{width - 44}" height="88" rx="10" fill="#151a24" stroke="#293140"/>'
            body += f'<text x="38" y="{y + 18}" fill="#8fa9c4" font-size="11" font-weight="600">{escape(lineage)}</text>'
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
                dash = ' stroke-dasharray="5 4"' if restored else ""
                color = "#ef8398" if restored else "#8fa9c4"
                marker = "report-restore-arrow" if restored else "report-parent-arrow"
                if y1 == y2:
                    direction = 1 if x2 >= x1 else -1
                    start_x = x1 + direction * 102
                    end_x = x2 - direction * 102
                    bend = max(28, abs(end_x - start_x) / 2)
                    path = (
                        f"M{start_x:.1f} {y1:.1f} "
                        f"C{start_x + direction * bend:.1f} {y1:.1f},"
                        f"{end_x - direction * bend:.1f} {y2:.1f},"
                        f"{end_x:.1f} {y2:.1f}"
                    )
                else:
                    direction = 1 if y2 > y1 else -1
                    start_y = y1 + direction * 24
                    end_y = y2 - direction * 24
                    bend = max(30, abs(end_y - start_y) / 2)
                    path = (
                        f"M{x1:.1f} {start_y:.1f} "
                        f"C{x1:.1f} {start_y + direction * bend:.1f},"
                        f"{x2:.1f} {end_y - direction * bend:.1f},"
                        f"{x2:.1f} {end_y:.1f}"
                    )
                body += f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".8"{dash} marker-end="url(#{marker})"/>'
        for identity, stage in by_id.items():
            x, y = positions[identity]
            phase = stage.get("kind") == "lineage_start"
            label = str(stage.get("stage") or stage.get("lineage") or identity)
            detail = (
                f"{stage.get('worker', worker)} · {label}; lineage {stage.get('lineage') or 'unknown'}; "
                f"metric {stage.get('metric') or '—'}; parent {stage.get('parent') or '—'}; restored {stage.get('restored') or '—'}"
            )
            dash = ' stroke-dasharray="5 4"' if phase else ""
            fill = "#1b202b" if phase else "#202633"
            stroke = "#5d6575" if phase else "#455168"
            body += f'<g><title>{escape(detail)}</title><rect x="{x - 102:.1f}" y="{y - 24:.1f}" width="204" height="48" rx="9" fill="{fill}" stroke="{stroke}"{dash}/>'
            body += f'<rect x="{x - 102:.1f}" y="{y - 14:.1f}" width="3" height="28" rx="1.5" fill="#b8a4e3"/>'
            shown = label if len(label) <= 23 else label[:22] + "…"
            body += f'<text x="{x - 86:.1f}" y="{y + 5:.1f}" fill="#eee7dc" font-size="12" font-weight="600">{escape(shown)}</text></g>'
    return _frame("Stage lineage", body, height=height, width=width)


def related_work_svg(report: ReportDocument) -> str:
    values = [
        item
        for item in report.related_works
        if isinstance(item.get("year"), int) and isinstance(item.get("citations"), int)
    ]
    if not values:
        return _empty(
            "Related research landscape", "No related-work metadata was retrieved."
        )
    years = [int(item["year"]) for item in values]
    weights = [math.log1p(int(item["citations"])) for item in values]
    low_year, high_year = min(years), max(years)
    low_weight, high_weight = min(weights), max(weights)
    year_span = max(1, high_year - low_year)
    weight_span = max(1.0, high_weight - low_weight)
    body = (
        '<line x1="72" y1="64" x2="72" y2="225" stroke="#536575"/>'
        '<line x1="72" y1="225" x2="756" y2="225" stroke="#536575"/>'
        '<text x="72" y="250" fill="#9ca8b8" font-size="12">publication year</text>'
        '<text x="76" y="82" fill="#9ca8b8" font-size="12">log citations</text>'
    )
    for index, item in enumerate(values):
        x = 92 + (int(item["year"]) - low_year) / year_span * 635
        y = 210 - (math.log1p(int(item["citations"])) - low_weight) / weight_span * 125
        radius = 7 + min(11, math.sqrt(max(0, int(item["citations"]))) * 0.55)
        color = _COLORS[index % len(_COLORS)]
        title = escape(
            f"{item['id']}: {item['title']} ({item['year']}; cited {item['citations']})"
        )
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity=".72"><title>{title}</title></circle>'
        body += f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" fill="#111621" font-size="9" font-weight="700">{escape(str(item["id"]))}</text>'
    body += f'<text x="92" y="242" fill="#9ca8b8" font-size="10">{low_year}</text><text x="727" y="242" fill="#9ca8b8" font-size="10">{high_year}</text>'
    return _frame("Related research landscape", body)


def resource_svg(report: ReportDocument) -> str:
    values = [
        (key, value) for key, value in report.eec.items() if isinstance(value, int)
    ]
    if not values:
        return _empty("Execution control events", "No EEC events were recorded.")
    values = values[:8]
    maximum = max((value for _, value in values), default=1) or 1
    body = ""
    for index, (label, value) in enumerate(values):
        y = 64 + index * 27
        width = value / maximum * 490
        body += f'<text x="28" y="{y + 13}" fill="#9ca8b8" font-size="11">{escape(label[:22])}</text>'
        body += f'<rect x="230" y="{y}" width="{width:.1f}" height="16" rx="4" fill="{_COLORS[index % len(_COLORS)]}"/>'
        body += f'<text x="{240 + width:.1f}" y="{y + 13}" fill="#eee7dc" font-size="11">{value}</text>'
    return _frame(
        "Execution control events", body, height=max(160, 86 + len(values) * 27)
    )


def finalist_svg(report: ReportDocument) -> str:
    values = [
        value
        for value in report.finalists
        if isinstance(value.get("metric"), (int, float))
    ]
    if not values:
        return _empty(
            "Final candidate comparison", "No evaluated finalists were recorded."
        )
    metrics = [float(value["metric"]) for value in values]
    low, high = min(metrics), max(metrics)
    span = max(high - low, max(abs(low), abs(high), 1) * 0.02)
    baseline = low - span * 0.08
    colors = ("#b8a4e3", "#8fa9c4", "#d8be91", "#87b6a7")
    body = (
        '<line x1="90" y1="220" x2="754" y2="220" stroke="#536575"/>'
        f'<text x="90" y="246" fill="#9ca8b8" font-size="12">{escape(report.metric_name)}</text>'
    )
    slot = 620 / max(1, len(values))
    for index, value in enumerate(values):
        metric = float(value["metric"])
        height = max(8, (metric - baseline) / max(high - baseline, 1e-12) * 135)
        x = 112 + index * slot
        width = min(86, slot * 0.55)
        y = 220 - height
        color = colors[index % len(colors)]
        body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="5" fill="{color}"/>'
        body += f'<text x="{x + width / 2:.1f}" y="{y - 9:.1f}" text-anchor="middle" fill="#eee7dc" font-size="12">{metric:.6g}</text>'
        body += f'<text x="{x + width / 2:.1f}" y="238" text-anchor="middle" fill="#9ca8b8" font-size="11">{escape(str(value["candidate_id"]))}</text>'
    return _frame("Final candidate comparison", body)


def artifact_svg(report: ReportDocument) -> str:
    if report.artifact_preview.get("kind") != "circle_packing":
        return _empty(
            "Best result artifact",
            "No structured visual preview is available for this artifact type.",
        )
    circles = report.artifact_preview.get("circles") or []
    if not circles:
        return _empty("Best result artifact", "The circle layout was empty.")
    body = (
        '<rect x="170" y="55" width="460" height="460" fill="#10121c" stroke="#8fa9c4" stroke-width="2"/>'
        '<text x="170" y="540" fill="#9ca8b8" font-size="12">Unit-square layout from the highest-ranked validated finalist</text>'
    )
    colors = ("#8fa9c4", "#b8a4e3", "#d8be91", "#87b6a7")
    for index, circle in enumerate(circles):
        x, y, radius = (float(value) for value in circle)
        cx = 170 + x * 460
        cy = 515 - y * 460
        r = max(0, radius * 460)
        color = colors[index % len(colors)]
        body += f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="1.5"/>'
    return _frame("Best result artifact", body, height=570)


def report_svgs(report: ReportDocument) -> dict[str, str]:
    return {
        "metric-history.svg": metric_svg(report),
        "candidate-comparison.svg": finalist_svg(report),
        "stage-lineage.svg": lineage_svg(report),
        "resource-timeline.svg": resource_svg(report),
        "result-artifact.svg": artifact_svg(report),
        "related-work.svg": related_work_svg(report),
    }
