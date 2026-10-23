// Lineage lanes: each Lxx gets a row; stages advance left to right.
function layoutLineage(nodes, byId, selected) {
  const local = new Map(nodes.map((node) => [node.id, node]));
  const lanes = new Map();
  for (const node of nodes) {
    const lineage = node.lineage || "Unrecorded";
    if (!lanes.has(lineage)) lanes.set(lineage, []);
    lanes.get(lineage).push(node);
  }
  const order = (a, b) =>
    (a.timestamp ?? Number.MAX_SAFE_INTEGER) -
      (b.timestamp ?? Number.MAX_SAFE_INTEGER) ||
    String(a.id).localeCompare(String(b.id));
  for (const members of lanes.values()) members.sort(order);
  const positions = new Map(),
    placing = new Set();
  const laneNames = [...lanes.keys()],
    laneIndex = new Map(laneNames.map((name, index) => [name, index]));
  function place(lineage) {
    if (placing.has(lineage) || positions.has(lanes.get(lineage)[0].id)) return;
    placing.add(lineage);
    const members = lanes.get(lineage),
      first = members[0],
      sourceId = first.restored || first.parent,
      source = local.get(sourceId),
      sourceLineage = source?.lineage || "Unrecorded";
    if (source && sourceLineage !== lineage && lanes.has(sourceLineage))
      place(sourceLineage);
    const origin = source ? positions.get(source.id) : null,
      start = origin ? origin.x + 102 : 150,
      y = 46 + laneIndex.get(lineage) * 132;
    members.forEach((node, index) =>
      positions.set(node.id, { x: start + index * 250 - 102, y }),
    );
    placing.delete(lineage);
  }
  laneNames.forEach(place);
  const laneRects = laneNames.map((lineage, index) => ({
    lineage,
    y: 24 + index * 132,
    height: 112,
  }));
  const width = Math.max(
      1000,
      ...[...positions.values()].map((point) => point.x + 244),
    ),
    height = Math.max(180, 34 + laneNames.length * 132);
  return { positions, laneRects, width, height };
}
