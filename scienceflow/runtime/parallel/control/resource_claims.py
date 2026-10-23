"""Conservative cross-run resource reservations under the managed registry lock."""

import os

from scienceflow.runtime.core.support.system_resources import (
    parse_cpu_list,
    parse_gpu_list_auto,
    query_gpu_devices,
)


class ResourceBusy(ValueError):
    pass


def claims_for(draft):
    cpu = set(parse_cpu_list(str(draft.get("cpu_list") or "")))
    raw = str(draft.get("gpu_list") or "").lower()
    gpu = (
        set()
        if raw in {"cpu", "none", "off", "disabled", "-1", ""}
        else (
            {"*"} if raw.startswith("auto") else {str(i) for i in parse_cpu_list(raw)}
        )
    )
    return {"cpu": sorted(cpu), "gpu": sorted(gpu)}


def reserve_resources(draft, rows):
    """Choose concrete GPUs for auto requests and reject overlapping pools."""
    claims = claims_for(draft)
    occupied = [
        (r, r.get("resource_claims") or claims_for(r.get("draft") or {}))
        for r in rows
        if r["alive"]
    ]
    if "*" in claims["gpu"]:
        raw = str(draft.get("gpu_list") or "auto")
        _, count, candidates = parse_gpu_list_auto(raw)
        if raw.strip().lower() == "auto":
            count = max(count, int(draft.get("workers") or 2))
        if count < 1:
            raise ResourceBusy("GPU count must be positive.")
        busy = {gpu for _, c in occupied for gpu in c["gpu"]}
        free = (
            []
            if "*" in busy
            else [
                str(g.index)
                for g in query_gpu_devices()
                if str(g.index) not in busy
                and (not candidates or g.index in candidates)
            ]
        )
        if len(free) < count:
            raise ResourceBusy(
                "Not enough unreserved GPUs. Choose another GPU pool or retry after a task finishes."
            )
        claims["gpu"] = free[:count]
    affinity = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    if affinity is not None and not set(claims["cpu"]).issubset(affinity):
        raise ResourceBusy(
            "The requested CPU pool is outside the current process affinity."
        )
    for row, other in occupied:
        cpu = set(claims["cpu"]) & set(other["cpu"])
        gpu = set(claims["gpu"]) & set(other["gpu"])
        if claims["gpu"] and "*" in other["gpu"]:
            gpu = set(claims["gpu"])
        if cpu or gpu:
            path = row.get("control_workspace") or row.get("cwd", "")
            raise ResourceBusy(
                f"Resources reserved by {path}: CPU {sorted(cpu)}; GPU {sorted(gpu)}. "
                "Change this task’s pool or retry after the other task stops."
            )
    return claims


def task_requests(tasks):
    """Use resolved runner inputs for both single-task and batch reservations."""
    return [
        dict(
            cpu_list=task.cpu_list,
            gpu_list=task.gpu_list_raw,
            workers=(task.lnr_patch or {}).get("num_workers", 2),
        )
        for task in tasks
    ]


def apply_reservation(record, rows):
    requests = record.get("resource_requests") or [record.get("draft") or {}]
    claims = [reserve_resources(request, rows) for request in requests]
    record["resource_claims"] = {
        kind: sorted({value for claim in claims for value in claim[kind]})
        for kind in ("cpu", "gpu")
    }
    tasks = record["manifest_payload"].get("tasks", [])
    for task, request, claim in zip(tasks, requests, claims):
        if str(request.get("gpu_list", "")).startswith("auto"):
            task["gpu_list"] = ",".join(claim["gpu"])
    if len(claims) == 1 and record.get("draft"):
        if str(requests[0].get("gpu_list", "")).startswith("auto"):
            record["draft"]["gpu_list"] = ",".join(claims[0]["gpu"])
