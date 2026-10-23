"""Bounded report finalization, independent of worker research time."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import sys


def report_budget(payload: dict) -> float:
    value = payload.get("report_budget_sec", 300)
    if isinstance(value, bool):
        raise ValueError("report_budget_sec must be a positive finite number")
    try:
        budget = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("report_budget_sec must be a positive finite number") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("report_budget_sec must be a positive finite number")
    return budget


async def finalize_reports(record: dict) -> tuple[list[dict], bool]:
    """Reap the report process at its own deadline; inspect committed files."""
    budget = report_budget(record.get("manifest_payload") or {})
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        __name__,
        record["run_id"],
        record["launch_token"],
        str(os.getpid()),
        str(budget),
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=budget)
        timed_out = process.returncode == -signal.SIGALRM
    except (TimeoutError, asyncio.CancelledError):
        timed_out = True
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    from scienceflow.research.reporting import read_report_manifest
    from scienceflow.runtime.parallel.location.local import validate_record_location

    validate_record_location(record)
    results = []
    for root in record.get("task_roots") or ():
        manifest = read_report_manifest(root)
        if (
            not manifest
            or manifest.get("run_id") != record["run_id"]
            or manifest.get("attempt") != record["attempt"]
            or manifest.get("status") != "ready"
        ):
            manifest = {"status": "generation_failed"}
        results.append(dict(manifest, task_root=root))
    return results, timed_out
