"""Renderer-neutral helpers for recorded stage ancestry graphs."""

from __future__ import annotations

from typing import Any


def lineage_depths(nodes: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Return top-down DAG ranks while tolerating malformed historical cycles."""

    depths: dict[str, int] = {}

    def visit(identity: str, ancestors: frozenset[str]) -> int:
        if identity in depths:
            return depths[identity]
        if identity in ancestors:
            return 0
        stage = nodes[identity]
        parents = [
            str(value)
            for value in (stage.get("parent"), stage.get("restored"))
            if value and str(value) in nodes and str(value) != identity
        ]
        lineage = ancestors | {identity}
        value = 1 + max((visit(parent, lineage) for parent in parents), default=-1)
        depths[identity] = value
        return value

    for identity in nodes:
        visit(identity, frozenset())
    return depths


def lineage_layout(
    nodes: dict[str, dict[str, Any]],
    *,
    x_start: float = 150,
    x_step: float = 250,
) -> tuple[dict[str, tuple[float, int]], float, list[str]]:
    """Place every lineage on one row and advance its stages left to right."""

    lanes: dict[str, list[str]] = {}
    for identity, node in nodes.items():
        lineage = str(node.get("lineage") or "Unrecorded")
        lanes.setdefault(lineage, []).append(identity)

    def order(identity: str) -> tuple[float, str]:
        value = nodes[identity].get("timestamp")
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            timestamp = float("inf")
        return timestamp, identity

    for values in lanes.values():
        values.sort(key=order)
    lane_names = list(lanes)
    lane_index = {lineage: index for index, lineage in enumerate(lane_names)}
    positions: dict[str, tuple[float, int]] = {}
    placing: set[str] = set()

    def place(lineage: str) -> None:
        members = lanes[lineage]
        if not members or members[0] in positions or lineage in placing:
            return
        placing.add(lineage)
        first = nodes[members[0]]
        source_id = next(
            (
                str(value)
                for value in (first.get("restored"), first.get("parent"))
                if value and str(value) in nodes and str(value) != members[0]
            ),
            "",
        )
        if source_id:
            source_lineage = str(nodes[source_id].get("lineage") or "Unrecorded")
            if source_lineage != lineage and source_lineage in lanes:
                place(source_lineage)
        start = positions.get(source_id, (x_start, 0))[0]
        for index, identity in enumerate(members):
            positions[identity] = (start + index * x_step, lane_index[lineage])
        placing.remove(lineage)

    for lineage in lane_names:
        place(lineage)
    width = max(
        1000.0,
        max((x for x, _ in positions.values()), default=x_start) + 150,
    )
    return positions, width, lane_names


__all__ = ["lineage_depths", "lineage_layout"]
