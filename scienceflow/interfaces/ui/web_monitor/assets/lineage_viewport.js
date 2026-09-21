// Per-worker viewport state and input gestures, independent of graph drawing.
const lineageZoom = new Map(),
  lineageScroll = new Map();
let lineageDragging = false;
function createLineageViewport(box, { width, height }, selected, taskKey) {
  const toolbar = add(box, "div", undefined, "zoom-toolbar");
  const minus = add(toolbar, "button", "−"),
    plus = add(toolbar, "button", "+"),
    fit = add(toolbar, "button", "Fit"),
    actual = add(toolbar, "button", "100%");
  const range = add(toolbar, "input");
  range.type = "range";
  range.min = "1";
  range.max = "200";
  range.setAttribute("aria-label", "Lineage zoom");
  const label = add(toolbar, "small");
  const scroll = add(
    box,
    "div",
    undefined,
    "scroll chart-surface lineage-scroll",
  );
  scroll.dataset.scrollKey = "lineage-" + selected;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "lineage-plot",
    role: "img",
    "aria-label":
      "Stage ancestry for " + selected + " grouped by recorded lineage",
  });
  scroll.append(svg);
  const zoomKey = taskKey + "/" + selected;
  const fitted = () =>
    Math.min(
      1,
      Math.max(0.01, ((scroll.clientWidth || 1000) - 18) / width),
      Math.max(0.01, 460 / height),
    );
  function zoom(value, anchor = { x: 0, y: 0 }) {
    const old = Number(svg.dataset.zoom) || 1,
      z = Math.max(0.01, Math.min(2, value));
    const left = ((scroll.scrollLeft + anchor.x) * z) / old - anchor.x,
      top = ((scroll.scrollTop + anchor.y) * z) / old - anchor.y;
    svg.style.width = width * z + "px";
    svg.style.height = height * z + "px";
    svg.dataset.zoom = z;
    range.value = String(z * 100);
    label.textContent = Math.round(z * 100) + "%";
    scroll.scrollLeft = left;
    scroll.scrollTop = top;
    lineageZoom.set(zoomKey, z);
    lineageScroll.set(zoomKey, {
      top: scroll.scrollTop,
      left: scroll.scrollLeft,
    });
  }
  const saved = lineageScroll.get(zoomKey);
  zoom(lineageZoom.get(zoomKey) ?? fitted());
  if (saved) {
    scroll.scrollTop = saved.top;
    scroll.scrollLeft = saved.left;
  }
  scroll.onscroll = () =>
    lineageScroll.set(zoomKey, {
      top: scroll.scrollTop,
      left: scroll.scrollLeft,
    });
  minus.onclick = () => zoom(Number(svg.dataset.zoom) / 1.25);
  plus.onclick = () => zoom(Number(svg.dataset.zoom) * 1.25);
  fit.onclick = () => zoom(fitted());
  actual.onclick = () => zoom(1);
  range.oninput = () => zoom(Number(range.value) / 100);
  scroll.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = scroll.getBoundingClientRect();
      const delta =
        event.deltaY *
        (event.deltaMode === 1
          ? 16
          : event.deltaMode === 2
            ? scroll.clientHeight
            : 1);
      zoom(
        Number(svg.dataset.zoom) *
          Math.exp(-Math.max(-200, Math.min(200, delta)) * 0.002),
        {
          x: event.clientX - rect.left - scroll.clientLeft - 8,
          y: event.clientY - rect.top - scroll.clientTop - 8,
        },
      );
    },
    { passive: false },
  );
  let drag = null;
  scroll.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest(".lineage-node")) return;
    const rect = scroll.getBoundingClientRect();
    if (
      event.clientX >= rect.left + scroll.clientLeft + scroll.clientWidth ||
      event.clientY >= rect.top + scroll.clientTop + scroll.clientHeight
    )
      return;
    drag = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      left: scroll.scrollLeft,
      top: scroll.scrollTop,
    };
    scroll.setPointerCapture(event.pointerId);
    scroll.classList.add("dragging");
    lineageDragging = true;
    event.preventDefault();
  });
  scroll.addEventListener("pointermove", (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    scroll.scrollLeft = drag.left + drag.x - event.clientX;
    scroll.scrollTop = drag.top + drag.y - event.clientY;
    lineageScroll.set(zoomKey, {
      top: scroll.scrollTop,
      left: scroll.scrollLeft,
    });
  });
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.id) return;
    drag = null;
    lineageDragging = false;
    scroll.classList.remove("dragging");
    if (scroll.hasPointerCapture(event.pointerId))
      scroll.releasePointerCapture(event.pointerId);
  }
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"])
    scroll.addEventListener(type, endDrag);
  add(toolbar, "small", "Wheel to zoom · Drag background to pan", "muted");
  return svg;
}
