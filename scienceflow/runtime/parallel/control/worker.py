"""Detached supervisor entry point; no terminal or interactive CLI ownership."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

from scienceflow.runtime.parallel.service import run_manifest
from scienceflow.research.solver.lnr.lifecycle.records.storage_usage import (
    record_final_storage,
)

from .registry import locked, read_record, run_dir, write_record


def _input_data_paths(record: dict) -> tuple[str, ...]:
    """Collect configured read-only inputs without touching their contents."""
    values: list[str] = []
    draft = record.get("draft")
    if isinstance(draft, dict):
        values.append(str(draft.get("input_data_dir") or ""))
    manifest = record.get("manifest_payload")
    if isinstance(manifest, dict):
        defaults = manifest.get("defaults")
        if isinstance(defaults, dict):
            values.append(
                str(
                    defaults.get("input_data_dir")
                    or defaults.get("data_dir")
                    or ""
                )
            )
        tasks = manifest.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    values.append(
                        str(
                            task.get("input_data_dir")
                            or task.get("data_dir")
                            or ""
                        )
                    )
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _record_terminal_storage(record: dict) -> None:
    excluded_paths = _input_data_paths(record)
    for root in record.get("task_roots") or ():
        try:
            record_final_storage(
                task_root_dir=root,
                excluded_paths=excluded_paths,
            )
        except Exception:  # noqa: BLE001
            # Telemetry is ancillary and cannot change the research outcome.
            continue


async def supervise(run_id: str, token: str) -> None:
    directory = run_dir(run_id)
    current = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, current.cancel)
    with locked(directory):
        record = read_record(directory)
        if record['launch_token'] != token or record.get('pid') != os.getpid():
            return
        if record['status'] == 'stopping':
            current.cancel()
        else:
            record['status'] = 'running'
        write_record(directory, record)
    status, error = 'completed', None
    try:
        summary = await run_manifest(record['manifest'])
        print(summary.text, flush=True)
        if any(r.status not in {'success', 'skipped'} for r in summary.results):
            status = 'failed'
    except asyncio.CancelledError:
        status = 'stopped'
    except Exception as exc:
        status, error = 'failed', type(exc).__name__
        print(f'Supervisor failed: {type(exc).__name__}', flush=True)
    finally:
        with locked(directory):
            record = read_record(directory)
            if record['launch_token'] == token:
                record.update(status=status, finished_at=time.time(), error=error)
                write_record(directory, record)

        if status == 'completed':
            from scienceflow.research.onboarding.library.export import export_success
            try:
                registration = await asyncio.to_thread(export_success, record)
            except Exception as exc:
                # Export is ancillary; it cannot turn completed research into a failure.
                registration = {'status': 'deferred', 'reason': f'Task export failed: {type(exc).__name__}'}
            with locked(directory):
                current_record = read_record(directory)
                if current_record['launch_token'] == token:
                    current_record['task_registration'] = registration
                    write_record(directory, current_record)
            print('Task registration: ' + registration['status'], flush=True)

        from scienceflow.research.reporting import generate_run_reports

        with locked(directory):
            current_record = read_record(directory)
        reports = await asyncio.to_thread(generate_run_reports, current_record)
        await asyncio.to_thread(_record_terminal_storage, current_record)
        public_reports = [
            {key: value for key, value in report.items() if key != 'task_root'}
            for report in reports
        ]
        with locked(directory):
            current_record = read_record(directory)
            if current_record['launch_token'] == token:
                current_record['final_reports'] = public_reports
                write_record(directory, current_record)
        print(
            'Final report: '
            + ', '.join(report.get('status', 'unknown') for report in reports),
            flush=True,
        )


if __name__ == '__main__':
    asyncio.run(supervise(sys.argv[1], sys.argv[2]))
