// One in-flight request chain and one timer, even during rapid navigation.
let busy = false,
  timer,
  refreshRequested = false,
  initialRefresh = true;
function scheduleRefresh(delay) {
  clearTimeout(timer);
  timer = setTimeout(refresh, delay);
}
function refreshSoon() {
  if (busy) refreshRequested = true;
  else scheduleRefresh(0);
}
async function refresh() {
  if (busy) {
    refreshRequested = true;
    return;
  }
  busy = true;
  const route = currentTaskRoute(),
    key = route.key,
    mode = route.mode;
  let delay = 10000;
  try {
    if (!document.hidden) {
      const fresh = initialRefresh ? "?fresh=1" : "";
      const response = await fetch(base + "api" + fresh, {
        signal: AbortSignal.timeout(initialRefresh ? 30000 : 8000),
      });
      if (!response.ok) throw Error("Monitor unavailable");
      const data = await response.json();
      if (Number.isFinite(data.refresh_sec))
        delay = Math.max(10000, data.refresh_sec * 1000);
      let detail = null;
      let report = null;
      if (key && mode === "monitor") {
        const response = await fetch(base + "task/" + key + fresh, {
          signal: AbortSignal.timeout(initialRefresh ? 30000 : 8000),
        });
        if (!response.ok) throw Error("Task unavailable");
        detail = await response.json();
      }
      if (key) {
        const response = await fetch(base + "task/" + key + "/report", {
          signal: AbortSignal.timeout(initialRefresh ? 30000 : 8000),
        });
        if (!response.ok) throw Error("Report unavailable");
        report = await response.json();
      }
      initialRefresh = false;
      const current = currentTaskRoute();
      if (key !== current.key || mode !== current.mode) return;
      // Scan timestamps change every cycle; they must not reset the entire view.
      const fingerprint = JSON.stringify([
        data.tasks,
        data.resources,
        detail,
        report,
        key,
        mode,
        taskPage,
        workerPage,
      ]);
      const editing = document.activeElement?.matches(
        ".worker-search, .zoom-toolbar input",
      );
      if (fingerprint !== last && !lineageDragging && !editing) {
        renderPreservingScroll(data, detail, key, report, mode);
        last = fingerprint;
      }
      document.querySelector("#fresh").textContent = data.updated_at
        ? `Updated ${new Date(data.updated_at * 1000).toLocaleTimeString()} · scan ${data.scan_ms}ms · background refresh`
        : "Loading initial snapshot…";
      document.querySelector("#host").textContent = key
        ? ""
        : data.resources || "";
      document.querySelector("#host").hidden = !!key;
      document.querySelector("#error").textContent = data.error
        ? `Refresh delayed (${data.error}); showing last snapshot`
        : "";
      if (mode === "report" && report?.available) delay = null;
    }
  } catch {
    document.querySelector("#error").textContent =
      "Refresh unavailable; retaining previous data. Retrying…";
  } finally {
    busy = false;
    if (refreshRequested) scheduleRefresh(0);
    else if (delay !== null) scheduleRefresh(delay);
    refreshRequested = false;
  }
}
window.addEventListener("hashchange", () => {
  lineageDragging = false;
  workerPage = 0;
  last = "";
  refreshSoon();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshSoon();
});
refresh();
