const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { JSDOM } = require("jsdom");

const root = join(__dirname, "../../scienceflow/interfaces/ui/web_monitor");
const assets = join(root, "assets");
// Read the assembler order so these tests execute the same scripts as PAGE.
const order = [
  ...readFileSync(join(root, "page.py"), "utf8")
    .match(/_SCRIPTS = \(([\s\S]*?)\)/)[1]
    .matchAll(/"([^"]+\.js)"/g),
].map((m) => m[1]);
const source = order
  .map((name) => readFileSync(join(assets, name), "utf8"))
  .join("\n")
  .replace(/refresh\(\);\s*$/, "");
const tasks = {
  tasks: [{ key: "1", name: "Circle packing", workers: ["w00", "w01"] }],
  resources: "CPU 8",
  updated_at: 1,
};
const trace = {
  started_at: 1000,
  points: [
    { metric: 0, timestamp: 1000, valid_comparable: true },
    { metric: 1, timestamp: 4600, valid_comparable: true },
  ],
  lineage: [
    { id: "a", worker: "w00", stage: "S01", lineage: "L01" },
    { id: "b", worker: "w00", stage: "S02", lineage: "L01", parent: "a" },
    { id: "c", worker: "w00", stage: "S03", lineage: "L02", restored: "b" },
    { id: "d", worker: "w01", stage: "S04", lineage: "L01", restored: "c" },
  ],
};
function fixture(t) {
  const dom = new JSDOM(
    readFileSync(join(assets, "shell.html"), "utf8").replace(
      "__SCIENCEFLOW_CSS__",
      "",
    ),
    {
      runScripts: "outside-only",
      url: "http://localhost/capability/#task/1",
      pretendToBeVisual: true,
    },
  );
  t.after(() => dom.window.close());
  const w = dom.window,
    timers = new Map();
  let sequence = 0;
  w.setTimeout = (callback, delay) => {
    timers.set(++sequence, { callback, delay });
    return sequence;
  };
  w.clearTimeout = (id) => timers.delete(id);
  w.AbortSignal.timeout = () => new w.AbortController().signal;
  w.fetch = async (url) => ({
    ok: true,
    json: async () => (url.includes("api") ? tasks : trace),
  });
  w.eval(
    `(()=>{'use strict';\n${source}\nwindow.testMonitor={renderPreservingScroll,refresh,refreshSoon,layoutLineage};})();`,
  );
  const api = w.testMonitor,
    doc = w.document;
  const render = (data = tasks, detail = trace) =>
    api.renderPreservingScroll(data, detail, "1");
  const node = (id) =>
    [...doc.querySelectorAll(".lineage-node")].find(
      (n) => n.dataset.node === id,
    );
  const worker = (id) =>
    [...doc.querySelectorAll(".worker-tabs button")]
      .find((n) => n.textContent === id)
      .click();
  return { w, doc, api, timers, render, node, worker };
}
test("recorded lineage uses horizontal L lanes and separate worker pages", (t) => {
  const { render, doc, node, worker } = fixture(t);
  render();
  assert.deepEqual(
    [...doc.querySelectorAll(".lane-label")].map((n) => n.textContent),
    ["L01", "L02"],
  );
  assert.equal(
    node("b").querySelector("rect").getAttribute("x"),
    node("c").querySelector("rect").getAttribute("x"),
  );
  assert(!node("d"));
  worker("w01");
  assert(node("d"));
  assert(!node("a"));
  doc.querySelector(".source-links button").click();
  assert(node("c").classList.contains("located"));
  assert.deepEqual(
    [...doc.querySelectorAll("section>h2")].map((n) => n.textContent),
    [
      "Metric history",
      "Stage lineage",
      "Resources & execution control",
      "Event logs",
    ],
  );
});
test("whole graph remains available beyond the old page limit and cycles terminate", (t) => {
  const { render, doc } = fixture(t);
  const lineage = Array.from({ length: 50 }, (_, i) => ({
    id: String(i),
    worker: "w00",
    stage: `S${i}`,
    lineage: `L${Math.floor(i / 10)}`,
    parent: i ? String(i - 1) : "1",
  }));
  render(tasks, { lineage });
  assert.equal(doc.querySelectorAll(".lineage-node").length, 50);
});
test("metric positions follow elapsed time and never invent missing timestamps", (t) => {
  const { render, doc } = fixture(t);
  render(tasks, {
    ...trace,
    points: [
      { metric: 0, timestamp: 1000 },
      { metric: 1, timestamp: 4600 },
      { metric: 2, timestamp: 15400 },
    ],
  });
  const xs = [...doc.querySelectorAll(".metric-dot")].map((n) =>
    Number(n.getAttribute("cx")),
  );
  assert(Math.abs((xs[1] - xs[0]) / (xs[2] - xs[0]) - 0.25) < 1e-8);
  assert(doc.body.textContent.includes("Elapsed time (h)"));
  render(tasks, {
    ...trace,
    points: [
      { metric: 0, timestamp: 1000 },
      { metric: 1, timestamp: 173800 },
    ],
  });
  assert(doc.body.textContent.includes("Elapsed time (day)"));
  render(tasks, { points: [{ metric: 5 }], lineage: [] });
  assert.equal(doc.querySelectorAll(".metric-dot").length, 0);
});
test("wheel zoom anchors the cursor and worker viewports retain their own zoom", (t) => {
  const { render, doc, w, worker } = fixture(t);
  render();
  const pane = doc.querySelector(".lineage-scroll"),
    svg = pane.querySelector("svg");
  pane.scrollLeft = 100;
  const before = (100 + 192) / Number(svg.dataset.zoom);
  const event = new w.WheelEvent("wheel", {
    deltaY: -100,
    clientX: 200,
    clientY: 100,
    cancelable: true,
  });
  pane.dispatchEvent(event);
  assert(event.defaultPrevented);
  assert(
    Math.abs((pane.scrollLeft + 192) / Number(svg.dataset.zoom) - before) <
      1e-8,
  );
  const zoom = svg.dataset.zoom;
  worker("w01");
  worker("w00");
  assert.equal(doc.querySelector(".lineage-plot").dataset.zoom, zoom);
});
test("dragging moves the viewport and suspends refresh rendering until release", async (t) => {
  const { api, doc, w } = fixture(t);
  await api.refresh();
  const pane = doc.querySelector(".lineage-scroll");
  Object.defineProperty(pane, "clientWidth", { value: 1000 });
  Object.defineProperty(pane, "clientHeight", { value: 500 });
  pane.setPointerCapture = () => {};
  pane.hasPointerCapture = () => false;
  function pointer(type, x, y) {
    const event = new w.Event(type, { bubbles: true, cancelable: true });
    Object.assign(event, { pointerId: 1, button: 0, clientX: x, clientY: y });
    pane.dispatchEvent(event);
  }
  pointer("pointerdown", 100, 100);
  pointer("pointermove", 70, 80);
  assert.equal(pane.scrollLeft, 30);
  w.fetch = async (url) => ({
    ok: true,
    json: async () =>
      url.includes("api") ? { ...tasks, resources: "CPU 9" } : trace,
  });
  await api.refresh();
  assert.equal(doc.querySelector(".lineage-scroll"), pane);
  pointer("pointerup", 70, 80);
  await api.refresh();
  assert.notEqual(doc.querySelector(".lineage-scroll"), pane);
});
test("metadata-only refresh preserves DOM and updates retain expanded records and scroll", async (t) => {
  const { api, doc, w } = fixture(t);
  await api.refresh();
  const pane = doc.querySelector(".lineage-scroll");
  doc.querySelector("details").open = true;
  pane.scrollTop = 85;
  w.fetch = async (url) => ({
    ok: true,
    json: async () =>
      url.includes("api") ? { ...tasks, updated_at: 2 } : trace,
  });
  await api.refresh();
  assert.equal(doc.querySelector(".lineage-scroll"), pane);
  w.fetch = async (url) => ({
    ok: true,
    json: async () =>
      url.includes("api") ? { ...tasks, resources: "CPU 9" } : trace,
  });
  await api.refresh();
  assert(doc.querySelector("details").open);
  assert.equal(doc.querySelector(".lineage-scroll").scrollTop, 85);
});
test("rapid refresh requests keep a single timer and request chain", async (t) => {
  const { api, w, timers } = fixture(t);
  let resolve,
    calls = 0;
  w.fetch = async (url) => {
    calls++;
    return url.includes("api")
      ? new Promise((r) => (resolve = r))
      : { ok: true, json: async () => trace };
  };
  const pending = api.refresh();
  api.refreshSoon();
  api.refreshSoon();
  await api.refresh();
  assert.equal(calls, 1);
  assert.equal(timers.size, 0);
  resolve({ ok: true, json: async () => tasks });
  await pending;
  assert.equal(calls, 3);
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].delay, 0);
});
test("HTML-like task names remain text and detail failures retain the previous view", async (t) => {
  const { api, w, doc, render } = fixture(t);
  render({
    ...tasks,
    tasks: [{ ...tasks.tasks[0], name: "<img src=x onerror=alert(1)>" }],
  });
  assert.equal(doc.querySelectorAll("#view img").length, 0);
  const previous = doc.querySelector("#view").innerHTML;
  w.fetch = async (url) => ({
    ok: url.includes("api"),
    json: async () => tasks,
  });
  await api.refresh();
  assert.equal(doc.querySelector("#view").innerHTML, previous);
});
