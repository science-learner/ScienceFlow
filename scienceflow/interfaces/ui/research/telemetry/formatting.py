"""Compact device labels; GPU utilization and VRAM are device-wide readings."""


def percentage(value):
    return f"{value:.0f}%" if value is not None else "—"


def memory(used, total):
    return (
        f"{used / 1024:.1f}/{total / 1024:.0f}G"
        if used is not None and total is not None
        else "—"
    )


def device_ids(ids):
    if not ids:
        return "—"
    groups = []
    start = end = ids[0]
    for item in ids[1:] + [None]:
        if (
            item is not None
            and str(end).isdigit()
            and str(item).isdigit()
            and int(item) == int(end) + 1
        ):
            end = item
            continue
        groups.append(str(start) if start == end else f"{start}–{end}")
        start = end = item
    return ",".join(groups[:3]) + (
        f",+{len(groups) - 3} groups" if len(groups) > 3 else ""
    )


def gpu_summary(devices, *, compact=True):
    if not devices:
        return "GPU —"
    shared = " shared" if any(g.get("shared") for g in devices) else ""
    if len(devices) <= 2 or not compact:
        return (
            "GPU "
            + ", ".join(
                f"{g['id']}:{percentage(g.get('util'))} {memory(g.get('used'), g.get('total'))}"
                for g in devices
            )
            + " dev"
            + shared
        )
    utilization = [g.get("util") for g in devices]
    used, total = [g.get("used") for g in devices], [g.get("total") for g in devices]
    avg = (
        sum(utilization) / len(devices)
        if all(v is not None for v in utilization)
        else None
    )
    vram = (
        memory(sum(used), sum(total))
        if all(v is not None for v in used + total)
        else "—"
    )
    return f"GPU [{device_ids([g['id'] for g in devices])}] avg {percentage(avg)} · VRAM {vram} dev{shared}"


def worker_resources(sample, *, compact=True):
    cores = f"/{sample['cores']}c" if sample.get("cores") else ""
    return f"CPU {percentage(sample.get('cpu'))}{cores} · {gpu_summary(sample.get('gpus', []), compact=compact)}"
