const lineageWorkers = new Map();
function ancestry(detail) {
  const box = section("Stage lineage"),
    all = detail.lineage || [];
  if (!all.length) {
    add(
      box,
      "p",
      "Stage relationships will appear when ancestry is reported.",
      "empty",
    );
    return;
  }
  legend(box, [
    ["Parent → child", "#8fa9c4"],
    ["Restored from (dashed)", "#ef8398"],
  ]);
  const taskKey = location.hash || "overview";
  const workers = [...new Set(all.map((n) => n.worker || "Unknown"))].sort();
  let selected = lineageWorkers.get(taskKey);
  if (!workers.includes(selected)) selected = workers[0];
  lineageWorkers.set(taskKey, selected);
  const tabs = add(box, "div", undefined, "worker-tabs");
  tabs.setAttribute("role", "tablist");
  function switchWorker(worker, target) {
    const previous = box.querySelector(".lineage-scroll");
    if (previous)
      lineageScroll.set(taskKey + "/" + selected, {
        top: previous.scrollTop,
        left: previous.scrollLeft,
      });
    lineageWorkers.set(taskKey, worker);
    const marker = document.createComment("lineage");
    box.replaceWith(marker);
    ancestry(detail);
    const next = view.lastElementChild;
    marker.replaceWith(next);
    if (target) {
      const node = [...next.querySelectorAll(".lineage-node")].find(
        (n) => n.dataset.node === target,
      );
      if (node) {
        node.classList.add("located");
        const svg = next.querySelector("svg.lineage-plot"),
          pane = next.querySelector(".lineage-scroll"),
          rect = node.querySelector("rect"),
          z = Number(svg.dataset.zoom);
        pane.scrollLeft = Math.max(
          0,
          Number(rect.getAttribute("x")) * z - pane.clientWidth / 2,
        );
        pane.scrollTop = Math.max(
          0,
          Number(rect.getAttribute("y")) * z - pane.clientHeight / 2,
        );
      }
    }
  }
  if (workers.length > 8) {
    const search = add(box, "input", undefined, "worker-search");
    search.placeholder = "Find worker…";
    search.setAttribute("aria-label", "Find worker");
    search.oninput = () => {
      for (const button of tabs.children)
        button.hidden = !button.textContent
          .toLowerCase()
          .includes(search.value.toLowerCase());
    };
  }
  for (const worker of workers) {
    const button = add(tabs, "button", worker);
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(worker === selected));
    button.onclick = () => switchWorker(worker);
  }
  const nodes = all.filter((n) => (n.worker || "Unknown") === selected),
    byId = new Map(all.map((n) => [n.id, n]));
  const { positions, laneRects, width, height } = layoutLineage(
    nodes,
    byId,
    selected,
  );
  const svg = createLineageViewport(box, { width, height }, selected, taskKey);
  const defs = svgNode("defs");
  for (const [id, color] of [
    ["parent-arrow", "#8fa9c4"],
    ["restore-arrow", "#ef8398"],
  ]) {
    const marker = svgNode("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto-start-reverse",
    });
    marker.append(svgNode("path", { d: "M0 0 L10 5 L0 10Z", fill: color }));
    defs.append(marker);
  }
  svg.append(defs);
  for (const lane of laneRects)
    svg.append(
      svgNode("rect", {
        x: 8,
        y: lane.y,
        width: width - 16,
        height: lane.height,
        rx: 12,
        fill: "#151a24",
        stroke: "#293140",
      }),
      svgNode(
        "text",
        { x: 28, y: lane.y + 20, class: "lane-label" },
        lane.lineage,
      ),
    );
  for (const n of nodes) {
    const to = positions.get(n.id);
    for (const [id, restore] of [
      [n.parent, false],
      [n.restored, true],
    ]) {
      const from = positions.get(id);
      if (from && id !== n.id) {
        const sameLane = from.y === to.y,
          aligned = from.x === to.x,
          downward = to.y > from.y,
          x1 = sameLane ? from.x + 204 : aligned ? from.x + 102 : from.x + 204,
          y1 = sameLane ? from.y + 24 : downward ? from.y + 48 : from.y,
          x2 = sameLane ? to.x : aligned ? to.x + 102 : to.x,
          y2 = sameLane ? to.y + 24 : downward ? to.y : to.y + 48,
          bend = sameLane
            ? Math.max(28, (x2 - x1) / 2)
            : Math.max(30, Math.abs(y2 - y1) / 2),
          path = sameLane
            ? `M${x1} ${y1} C${x1 + bend} ${y1},${x2 - bend} ${y2},${x2} ${y2}`
            : `M${x1} ${y1} C${x1} ${y1 + (downward ? bend : -bend)},${x2} ${y2 + (downward ? -bend : bend)},${x2} ${y2}`;
        svg.append(
          svgNode("path", {
            d: path,
            stroke: restore ? "#ef8398" : "#8fa9c4",
            "stroke-width": 1.5,
            opacity: 0.8,
            fill: "none",
            "stroke-dasharray": restore ? "5 5" : "none",
            "marker-end": `url(#${restore ? "restore" : "parent"}-arrow)`,
          }),
        );
      }
    }
  }
  for (const n of nodes) {
    const p = positions.get(n.id),
      phaseOnly = n.kind === "lineage_start",
      group = svgNode("g", {
        class: `lineage-node${phaseOnly ? " lineage-start" : ""}`,
        tabindex: 0,
        "data-node": n.id,
      });
    group.append(
      svgNode(
        "title",
        {},
        phaseOnly
          ? `${n.worker} · ${n.lineage}\nLineage start · ${n.transition || "redirect"}\nStarted from ${n.restored || n.parent || "Not recorded"}\nPrevious lineage ${n.previous_lineage || "—"}`
          : `${n.worker} · ${n.stage}\nMetric ${metricValue(n.metric)} · Validation ${n.valid || "unknown"}\nLineage ${n.lineage || "unknown"}\nCandidate ${n.candidate || n.id}\nParent ${n.parent || n.parent_stage || "Not recorded"}\nRestored ${n.restored || "—"}`,
      ),
      svgNode("rect", {
        x: p.x,
        y: p.y,
        width: 204,
        height: 48,
        rx: 9,
        fill: phaseOnly ? "#1b202b" : "#202633",
        stroke: phaseOnly ? "#5d6575" : "#455168",
        "stroke-dasharray": phaseOnly ? "5 4" : "none",
      }),
      svgNode("rect", {
        x: p.x,
        y: p.y + 10,
        width: 3,
        height: 28,
        rx: 1.5,
        fill: "#b8a4e3",
      }),
    );
    const label = String(n.stage || n.lineage || n.id);
    group.append(
      svgNode(
        "text",
        { x: p.x + 16, y: p.y + 25, class: "node-title" },
        label.length > 23 ? label.slice(0, 22) + "…" : label,
      ),
    );
    svg.append(group);
  }
  const references = add(box, "div", undefined, "source-links");
  const seenSources = new Set();
  for (const n of nodes) {
    for (const id of [n.restored, n.parent]) {
      if (!id || positions.has(id) || seenSources.has(id)) continue;
      seenSources.add(id);
      const source = byId.get(id);
      if (source) {
        const button = add(
          references,
          "button",
          `Source: ${source.worker || "Unknown"} · ${source.stage || id}`,
        );
        button.onclick = () => switchWorker(source.worker || "Unknown", id);
      } else add(references, "small", `Source not recorded: ${id}`);
    }
  }
  add(
    box,
    "p",
    "Each lineage uses one row; stages advance left to right. New rows align with their recorded source stage.",
    "muted",
  );
  const details = add(box, "details", undefined, "lineage-details");
  add(details, "summary", "Relationship records");
  const table = add(add(details, "div", undefined, "scroll"), "table"),
    head = add(table, "tr");
  ["Stage / candidate", "Parent", "Restored from", "Lineage"].forEach((x) =>
    add(head, "th", x),
  );
  nodes.forEach((n) => {
    const r = add(table, "tr");
    [
      `${n.worker} ${n.stage || n.lineage} / ${n.id}`,
      n.parent || n.parent_stage || "Not recorded",
      n.restored || "—",
      n.lineage || "—",
    ].forEach((x) => add(r, "td", x));
  });
}
