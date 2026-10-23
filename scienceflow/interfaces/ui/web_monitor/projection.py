"""Pure Web trace projection; timestamps and explicit ancestry stay authoritative."""

import math
from bisect import bisect_right

from scienceflow.interfaces.ui.monitor_trace.presentation.builder import parse_row_time


def recorded_time(row) -> float | None:
    value = parse_row_time(row)
    return value if value is not None and math.isfinite(value) else None


_LINEAGE_TRANSITION_EVENTS = frozenset(
    {"estra_keep_current_compacted", "estra_stage_switched"}
)


def _event_payload(event):
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _lineage_sort_key(value):
    text = str(value or "")
    if len(text) > 1 and text[0].casefold() == "l" and text[1:].isdigit():
        return (0, int(text[1:]))
    return (1, text.casefold())


def lineage(rows, events=()):
    """Project formal stages and only transitions that reached that registry."""
    result = []
    boundaries = {}
    for event in events:
        stamp = recorded_time(event)
        if stamp is not None and event.get("event") == "estra_decision":
            payload = _event_payload(event)
            worker = event.get("worker_id") or payload.get("worker_id", "")
            worker = str(worker).casefold()
            boundaries.setdefault(worker, set()).add(stamp)
    boundaries = {worker: sorted(stamps) for worker, stamps in boundaries.items()}
    registered_lineages = set()
    for index, row in enumerate(rows):
        worker = str(row.get("worker_id") or "")
        lineage_id = str(row.get("lineage_id") or "").strip()
        if worker and lineage_id:
            registered_lineages.add((worker.casefold(), lineage_id))
        stamp = recorded_time(row)
        node_uid = row.get("node_uid")
        candidate = row.get("candidate_id")
        result.append(
            {
                "id": node_uid
                or candidate
                or f"{worker}:{row.get('stage_id', '')}:{row.get('row_order') or index}",
                "candidate": candidate or "",
                "worker": worker,
                "stage": row.get("stage_id", ""),
                "lineage": lineage_id,
                "parent": row.get("parent_node_uid")
                or row.get("parent_candidate_id")
                or "",
                "parent_stage": row.get("parent_stage_id", ""),
                "restored": row.get("restored_from_node_uid")
                or row.get("restored_from_candidate_id")
                or "",
                "metric": row.get("metric_value", ""),
                "valid": row.get("validation_ok", ""),
                "timestamp": stamp,
                "estra_phase": bisect_right(boundaries[worker.casefold()], stamp)
                if worker.casefold() in boundaries and stamp is not None
                else None,
                "kind": "evaluated_stage",
            }
        )
    # Older stage writers persisted ``parent_stage_id`` but not a node UID.
    # Resolve that relation within the same worker so renderers receive one
    # canonical graph edge instead of having to infer it independently.
    stage_nodes = {}
    for node in result:
        key = (str(node.get("worker") or "").casefold(), str(node.get("stage") or ""))
        if key[0] and key[1]:
            stage_nodes[key] = str(node["id"])
    for node in result:
        if node.get("parent") or not node.get("parent_stage"):
            continue
        node["parent"] = stage_nodes.get(
            (
                str(node.get("worker") or "").casefold(),
                str(node.get("parent_stage") or ""),
            ),
            "",
        )

    known_ids = {str(node["id"]) for node in result}
    transition_ids = set()
    for event in sorted(events, key=lambda value: recorded_time(value) or 0.0):
        event_name = str(event.get("event") or "")
        if event_name not in _LINEAGE_TRANSITION_EVENTS:
            continue
        payload = _event_payload(event)
        worker = str(event.get("worker_id") or payload.get("worker_id") or "")
        new_lineage = str(payload.get("new_lineage_id") or "").strip()
        if not worker or not new_lineage:
            continue
        if (worker.casefold(), new_lineage) not in registered_lineages:
            continue
        node_id = f"{worker}:{new_lineage}:phase"
        if node_id in known_ids or node_id in transition_ids:
            continue
        transition_ids.add(node_id)
        source_node_uid = str(payload.get("target_node_uid") or "")
        restored = event_name == "estra_stage_switched"
        result.append(
            {
                "id": node_id,
                "candidate": "",
                "worker": worker,
                "stage": "",
                "lineage": new_lineage,
                "parent": "" if restored else source_node_uid,
                "parent_stage": "",
                "restored": source_node_uid if restored else "",
                "metric": "",
                "valid": "",
                "timestamp": recorded_time(event),
                "estra_phase": None,
                "kind": "lineage_start",
                "previous_lineage": str(payload.get("previous_lineage_id") or ""),
                "transition": (
                    "restore" if event_name == "estra_stage_switched" else "redirect"
                ),
            }
        )
    transitions = [node for node in result if node.get("kind") == "lineage_start"]
    for transition in transitions:
        descendants = [
            node
            for node in result
            if node.get("kind") == "evaluated_stage"
            and str(node.get("worker") or "").casefold()
            == str(transition.get("worker") or "").casefold()
            and node.get("lineage") == transition.get("lineage")
        ]
        if not descendants:
            continue
        descendants.sort(key=lambda node: recorded_time(node) or 0.0)
        first = descendants[0]
        source = transition.get("restored") or transition.get("parent")
        if first.get("parent") == source or not first.get("parent"):
            first["parent"] = transition["id"]
        if first.get("restored") == source:
            first["restored"] = ""
    result.sort(
        key=lambda node: (
            _lineage_sort_key(node.get("lineage")),
            recorded_time(node) or 0.0,
            0 if node.get("kind") == "lineage_start" else 1,
        )
    )
    return result


def timestamped_points(points, rows):
    """Attach genuine timestamps even when a single-point trace uses row order."""
    row_times = {
        row.get("candidate_id") or row.get("node_uid"): recorded_time(row)
        for row in rows
        if row.get("candidate_id") or row.get("node_uid")
    }
    stage_times = {
        (
            str(row.get("worker_id") or "").casefold(),
            row.get("stage_id", ""),
        ): recorded_time(row)
        for row in rows
    }
    result = []
    for value in points:
        point = value.to_dict()
        stamp = row_times.get(point["candidate_id"])
        if stamp is None:
            stamp = stage_times.get((point["worker_id"].casefold(), point["stage_id"]))
        result.append(dict(point, timestamp=stamp))
    return result
