function plot(detail) {
  const box = section("Metric history");
  const points = (detail.points || [])
    .filter((p) => Number.isFinite(p.metric) && Number.isFinite(p.timestamp))
    .sort((a, b) => a.timestamp - b.timestamp);
  const origin = Number.isFinite(detail.started_at)
    ? Math.min(detail.started_at, points[0]?.timestamp ?? detail.started_at)
    : (points[0]?.timestamp ?? 0);
  const duration = points.length
      ? points[points.length - 1].timestamp - origin
      : 0,
    unit = duration >= 48 * 3600 ? "day" : "h",
    scale = unit === "day" ? 86400 : 3600,
    span = Math.max(duration, scale / 100);
  if (!points.length) {
    add(
      box,
      "p",
      "Waiting for evaluated stages with recorded timestamps.",
      "empty",
    );
    return;
  }
  const valid = points.filter((p) => p.valid_comparable);
  const best = valid.length
    ? valid.reduce((a, b) =>
        (detail.lower_is_better ? b.metric < a.metric : b.metric > a.metric)
          ? b
          : a,
      )
    : null;
  const meta = add(box, "div", undefined, "chart-meta");
  add(
    meta,
    "span",
    `${points.length} observations · ${detail.lower_is_better ? "Lower" : "Higher"} is better`,
    "muted",
  );
  add(meta, "strong", `Best valid ${best ? metricValue(best.metric) : "—"}`);
  legend(box, [
    ["Evaluated stage", "#8fa9c4"],
    ["Not comparable", "#ef8398"],
    ["Best valid so far", "#b8a4e3"],
  ]);
  const scroll = add(box, "div", undefined, "scroll chart-surface");
  scroll.dataset.scrollKey = "metric";
  const svg = svgNode("svg", {
    viewBox: "0 0 1040 340",
    class: "plot",
    role: "img",
    "aria-label": "Stage metric observations and best valid metric",
  });
  scroll.append(svg);
  const values = points.map((p) => p.metric);
  let lo = values.reduce((a, b) => Math.min(a, b)),
    hi = values.reduce((a, b) => Math.max(a, b));
  const margin = (hi - lo) * 0.12 || Math.max(Math.abs(lo) * 0.05, 0.01);
  lo -= margin;
  hi += margin;
  const x = (i) => 92 + ((points[i].timestamp - origin) * 908) / span,
    y = (v) => 278 - ((v - lo) * 232) / (hi - lo);
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4,
      yy = y(value);
    svg.append(
      svgNode("line", {
        x1: 92,
        y1: yy,
        x2: 1000,
        y2: yy,
        stroke: "#303747",
        "stroke-dasharray": "3 6",
      }),
      svgNode(
        "text",
        { x: 78, y: yy + 4, "text-anchor": "end", class: "axis-label" },
        metricValue(value),
      ),
    );
  }
  for (let i = 0; i <= 5; i++)
    svg.append(
      svgNode(
        "text",
        {
          x: 92 + (i * 908) / 5,
          y: 305,
          "text-anchor": "middle",
          class: "axis-label",
        },
        ((span * i) / 5 / scale).toFixed(2),
      ),
    );
  svg.append(
    svgNode(
      "text",
      { x: 546, y: 330, "text-anchor": "middle", class: "axis-label" },
      `Elapsed time (${unit})`,
    ),
  );
  let running = null,
    path = "";
  points.forEach((p, i) => {
    if (
      p.valid_comparable &&
      (running === null ||
        (detail.lower_is_better ? p.metric < running : p.metric > running))
    )
      running = p.metric;
    if (running !== null)
      path += path ? ` H${x(i)} V${y(running)}` : `M${x(i)} ${y(running)}`;
  });
  if (path)
    svg.append(
      svgNode("path", {
        d: path,
        fill: "none",
        stroke: "#b8a4e3",
        "stroke-width": 2.5,
        "stroke-linejoin": "round",
      }),
    );
  points.forEach((p, i) => {
    const isBest = p === best,
      dot = svgNode("circle", {
        cx: x(i),
        cy: y(p.metric),
        r: isBest ? 6 : 4,
        fill: p.valid_comparable ? "#8fa9c4" : "#ef8398",
        stroke: isBest ? "#eee7dc" : "#191e2a",
        "stroke-width": isBest ? 2 : 1,
        class: "metric-dot",
        tabindex: 0,
      });
    dot.append(
      svgNode(
        "title",
        {},
        `${p.worker_id || "Worker"} · ${p.stage_id || i + 1}\nElapsed ${((p.timestamp - origin) / scale).toFixed(2)} ${unit} · Metric ${metricValue(p.metric)} · ${p.valid_comparable ? "valid comparable" : "not comparable"}`,
      ),
    );
    svg.append(dot);
  });
}
