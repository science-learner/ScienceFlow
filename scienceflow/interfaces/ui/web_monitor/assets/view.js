function taskTabs(parent, key, mode) {
  const tabs = add(parent, "nav", undefined, "task-tabs");
  for (const [value, label] of [
    ["monitor", "Run monitor"],
    ["report", "Research report"],
  ]) {
    const link = add(tabs, "a", label, "task-tab");
    link.href = `#task/${key}/${value}`;
    link.setAttribute("aria-current", mode === value ? "page" : "false");
  }
}

function reportDownload(parent, key, format, label) {
  const link = add(parent, "a", label, "download-button");
  link.href = `${base}task/${key}/report.${format}`;
  link.download = `scienceflow-report.${format}`;
}

function renderReport(task, key, report) {
  const sectionNode = add(view, "section", undefined, "section report-view");
  add(sectionNode, "h2", "Scientific research report");
  if (!report?.available) {
    const active = report && report.terminal === false;
    add(
      sectionNode,
      "p",
      active
        ? "The evidence-backed research report will be generated after this task ends."
        : "No scientific report has been generated for this historical task yet.",
      "muted",
    );
    if (!active) {
      const button = add(
        sectionNode,
        "button",
        "Generate research report",
        "primary-button",
      );
      const message = add(sectionNode, "p", "", "muted");
      button.onclick = async () => {
        button.disabled = true;
        button.textContent = "Generating…";
        message.textContent =
          "Collecting evidence, related research, and report figures.";
        try {
          const response = await fetch(`${base}task/${key}/report`, {
            method: "POST",
            signal: AbortSignal.timeout(120000),
          });
          const value = await response.json();
          if (!response.ok) throw Error(value.error || "Report generation failed");
          last = "";
          refreshSoon();
        } catch (error) {
          button.disabled = false;
          button.textContent = "Retry research report";
          message.textContent = error.message || "Report generation failed";
          message.className = "error";
        }
      };
    }
    return;
  }
  const toolbar = add(sectionNode, "div", undefined, "report-toolbar");
  if (report.formats?.["report.md"])
    reportDownload(toolbar, key, "md", "Download Markdown");
  if (report.formats?.["report.html"])
    reportDownload(toolbar, key, "html", "Download HTML");
  if (report.formats?.["report.pdf"])
    reportDownload(toolbar, key, "pdf", "Download PDF");
  else add(toolbar, "span", "PDF unavailable", "muted");
  add(
    toolbar,
    "small",
    `Attempt ${report.attempt ?? "—"} · generated ${report.generated_at_utc ? new Date(report.generated_at_utc).toLocaleString() : "—"}`,
  );
  if (!report.formats?.["report.html"]) {
    add(sectionNode, "p", "The HTML report is unavailable.", "error");
    return;
  }
  const frame = add(sectionNode, "iframe", undefined, "report-frame");
  frame.title = `${task.name} final report`;
  frame.src = `${base}task/${key}/report.html`;
  frame.setAttribute("sandbox", "allow-same-origin");
  frame.onload = () => {
    try {
      frame.style.height = `${Math.max(720, frame.contentDocument.documentElement.scrollHeight + 24)}px`;
    } catch {}
  };
}

function renderMonitor(data, detail, task) {
  const stats = add(view, "div", undefined, "stats");
  stat(stats, task.metric || "Best", task.best);
  stat(
    stats,
    "Evaluated / valid",
    `${task.evaluated ?? "—"} / ${task.valid ?? "—"}`,
  );
  stat(
    stats,
    "ESTRA decisions / switches",
    `${task.estra?.estra_decision ?? "—"} / ${task.estra?.estra_stage_switched ?? "—"}`,
    "purple",
  );
  stat(stats, "EEC events", task.eec_total);
  stat(stats, "Cost USD", task.cost, "pink");
  add(view, "p", task.usage || "Usage awaiting reports", "muted");
  if (detail?.loading) {
    add(view, "p", "Loading curves and lineage in the background…", "muted");
  } else if (detail?.error) {
    add(view, "p", detail.error, "error");
  }
  plot(detail || {});
  ancestry(detail || {});
  workerPage = Math.min(
    workerPage,
    Math.max(0, Math.ceil((task.workers || []).length / 20) - 1),
  );
  const controls = section("Resources & execution control");
  add(
    controls,
    "p",
    `GPU current / GPU minutes: ${task.gpu ?? "—"} · Kills ${task.kills ?? "—"}`,
  );
  add(
    controls,
    "p",
    Object.entries(task.eec || {})
      .map(([k, v]) => k + " " + v)
      .join(" · ") || "No resource events reported",
  );
  add(
    controls,
    "p",
    `ESTRA compactions ${task.estra?.estra_keep_current_compacted ?? "—"} · Invalid ${task.estra?.estra_invalid ?? "—"}`,
  );
  add(controls, "h3", "Host resources");
  add(
    controls,
    "pre",
    data.resources || "Awaiting resource report",
    "log-scroll",
  ).dataset.scrollKey = "resources";
  add(controls, "h3", "Workers");
  pager(
    controls,
    (task.workers || []).length,
    20,
    workerPage,
    (page) => (workerPage = page),
  );
  add(
    controls,
    "pre",
    (task.workers || [])
      .slice(workerPage * 20, (workerPage + 1) * 20)
      .join("\n") || "No worker reports",
    "log-scroll",
  ).dataset.scrollKey = "workers";
  const events = section("Event logs");
  add(events, "h3", "Execution control");
  add(
    events,
    "pre",
    (task.recent || []).join("\n") || "No resource events reported",
    "log-scroll",
  ).dataset.scrollKey = "control-events";
  add(events, "h3", "Research trace");
  add(
    events,
    "pre",
    (detail?.events || [])
      .map((event) =>
        `${event.x_label || "Time unavailable"} · ${event.kind} · ${event.label}`,
      )
      .join("\n") || "No events reported",
    "log-scroll",
  ).dataset.scrollKey = "trace-events";
}

function render(data, detail, key, report, mode) {
  taskPage = Math.min(
    taskPage,
    Math.max(0, Math.ceil(data.tasks.length / 12) - 1),
  );
  view.replaceChildren();
  if (!key) {
    add(view, "h2", `Tasks · ${data.tasks.length}`);
    pager(view, data.tasks.length, 12, taskPage, (page) => (taskPage = page));
    const grid = add(view, "div", undefined, "grid");
    data.tasks.slice(taskPage * 12, (taskPage + 1) * 12).forEach((task) => {
      const card = add(grid, "div", undefined, "card");
      const link = add(card, "a", `${task.key} · ${task.name}`);
      link.href = `#task/${task.key}/monitor`;
      add(card, "p", task.status, "muted");
      const progress = add(card, "progress");
      progress.max = 1;
      progress.value = task.fraction || 0;
      add(card, "p", `Best ${task.best ?? "—"} · Cost ${task.cost ?? "—"}`);
      add(
        card,
        "small",
        `ESTRA ${task.estra?.estra_decision ?? "—"}/${task.estra?.estra_stage_switched ?? "—"} · EEC ${task.eec_total ?? "—"}`,
      );
    });
    return;
  }
  const task = data.tasks.find((candidate) => candidate.key === key);
  if (!task) {
    add(view, "p", "Task unavailable");
    return;
  }
  add(view, "h2", `${key} · ${task.name}`);
  add(view, "p", task.status, "muted");
  const progress = add(view, "progress");
  progress.max = 1;
  progress.value = task.fraction || 0;
  add(
    view,
    "p",
    `Time ${seconds(task.elapsed_sec)} / ${seconds(task.budget_sec)} · remaining ${seconds(task.budget_sec - task.elapsed_sec)}`,
  );
  taskTabs(view, key, mode);
  if (mode === "report") renderReport(task, key, report);
  else renderMonitor(data, detail, task);
}

function renderPreservingScroll(data, detail, key, report, mode) {
  const sameView =
    view.dataset.taskKey === (key || "") && view.dataset.taskMode === mode;
  const positions = new Map(
    [...view.querySelectorAll("[data-scroll-key]")].map((node) => [
      node.dataset.scrollKey,
      { top: node.scrollTop, left: node.scrollLeft },
    ]),
  );
  const expandedRecords = [...view.querySelectorAll("details")].map(
    (node) => node.open,
  );
  render(data, detail, key, report, mode);
  if (sameView)
    view
      .querySelectorAll("details")
      .forEach((node, index) => (node.open = expandedRecords[index] || false));
  if (sameView)
    for (const node of view.querySelectorAll("[data-scroll-key]")) {
      const position = positions.get(node.dataset.scrollKey);
      if (position) {
        node.scrollTop = position.top;
        node.scrollLeft = position.left;
      }
    }
  view.dataset.taskKey = key || "";
  view.dataset.taskMode = mode;
}
