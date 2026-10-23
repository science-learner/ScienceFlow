"""Run bounded evaluator samples before accepting a newly prepared contract."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


async def validate_evaluator(command: str, samples: dict, workspace: Path) -> dict:
    if '{artifact_abs_path}' not in command:
        raise ValueError('Prepared evaluator must accept {artifact_abs_path}.')
    paths = [Path(str(samples.get(key, ''))).expanduser() for key in ('valid_artifact', 'invalid_artifact')]
    if any(not p.is_absolute() or not p.is_file() for p in paths) or paths[0].resolve() == paths[1].resolve():
        raise ValueError('Provide separate existing valid and invalid sample artifact paths.')
    evidence = []
    # Deliberately use a different cwd, matching isolated worker execution.
    with TemporaryDirectory(prefix='scienceflow-evaluator-') as directory:
        for path in paths:
            rendered = command.replace('{artifact_abs_path}', shlex.quote(str(path))).replace('{python}', shlex.quote(sys.executable))
            result = await _run(rendered, Path(directory))
            evidence.append(result)
    valid, invalid = evidence
    try:
        metric = json.loads(valid['stdout'])['metric']
        good = isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric)
    except (ValueError, TypeError, KeyError):
        good = False
    if valid['exit_code'] != 0 or not good or invalid['exit_code'] == 0:
        raise ValueError('Evaluator validation failed: valid input must return a finite metric; invalid input must exit nonzero.')
    report = {'valid_passed': True, 'invalid_rejected': True, 'metric': metric,
              'command': command, 'samples': [str(p) for p in paths]}
    destination = workspace / '.scienceflow' / 'preparation' / 'validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


async def _run(command: str, cwd: Path) -> dict:
    process = await asyncio.create_subprocess_shell(command, cwd=cwd, start_new_session=True,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        async with asyncio.timeout(30):
            data = await process.stdout.read(65_537)
            if len(data) > 65_536:
                raise ValueError('Evaluator sample output exceeds 64 KiB.')
            await process.wait()
        return {'exit_code': process.returncode, 'stdout': data.decode('utf-8', errors='replace')}
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
