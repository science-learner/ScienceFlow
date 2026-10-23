const view = document.querySelector("#view"),
  base = location.pathname;
let taskPage = 0,
  workerPage = 0,
  last = "";
function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function add(parent, tag, text, cls) {
  const n = el(tag, text, cls);
  parent.append(n);
  return n;
}
function seconds(s) {
  s = Math.max(0, Math.floor(s || 0));
  return s >= 3600
    ? (s / 3600).toFixed(1) + "h"
    : s >= 60
      ? (s / 60).toFixed(1) + "m"
      : s + "s";
}
function pager(parent, total, size, page, set) {
  const nav = add(parent, "nav");
  const a = add(nav, "button", "Previous"),
    b = add(nav, "button", "Next");
  a.disabled = page === 0;
  b.disabled = (page + 1) * size >= total;
  add(
    nav,
    "small",
    `${Math.min(page * size + 1, total)}–${Math.min((page + 1) * size, total)} / ${total}`,
  );
  a.onclick = () => {
    set(page - 1);
    last = "";
    refreshSoon();
  };
  b.onclick = () => {
    set(page + 1);
    last = "";
    refreshSoon();
  };
}
function stat(parent, label, value, cls) {
  const n = add(parent, "div", undefined, "stat " + (cls || ""));
  add(n, "small", label);
  add(n, "div", value ?? "—", "value");
}
function section(title) {
  const n = add(view, "section", undefined, "section");
  add(n, "h2", title);
  return n;
}
function svgNode(tag, attrs = {}, text) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function metricValue(value) {
  return Number.isFinite(Number(value)) && value !== null && value !== ""
    ? Number(value).toLocaleString("en-US", { maximumSignificantDigits: 6 })
    : "—";
}
function legend(parent, items) {
  const row = add(parent, "div", undefined, "legend");
  for (const [label, color] of items) {
    const item = add(row, "span");
    const dot = add(item, "i");
    dot.style.background = color;
    item.append(document.createTextNode(label));
  }
}
function currentTaskRoute() {
  const match = location.hash.match(
    /^#task\/(\d+)(?:\/(monitor|report))?$/,
  );
  return match
    ? { key: match[1], mode: match[2] || "monitor" }
    : { key: null, mode: "monitor" };
}
