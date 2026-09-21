"""Evaluator interpreter overrides must preserve virtual environment identity."""

import json
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

from scienceflow.research.quality.evaluator.backends.command_env import (
    task_command_env,
    task_python_executable,
)


def test_explicit_python_preserves_real_virtual_environment(tmp_path):
    root = tmp_path / 'eval-env'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / 'bin/python'
    ctx = SimpleNamespace(cfg={'evaluator': {'command': {'python_executable': str(python)}}})
    selected = task_python_executable(ctx)
    assert selected == python
    env = task_command_env(ctx)
    assert env['VIRTUAL_ENV'] == str(root)
    result = subprocess.run(
        [str(selected), '-c', 'import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))'],
        env=env, check=True, capture_output=True, text=True,
    )
    prefix, base = json.loads(result.stdout)
    assert Path(prefix) == root
    assert prefix != base


def test_default_and_environment_path_remain_compatible(tmp_path):
    assert task_python_executable(SimpleNamespace(cfg={})) is None
    root = tmp_path / 'environment'
    (root / 'bin').mkdir(parents=True)
    (root / 'bin/python').symlink_to(sys.executable)
    ctx = SimpleNamespace(cfg={'evaluator': {'command': {'environment_path': str(root)}}})
    assert task_python_executable(ctx) == root / 'bin/python'
    ctx = SimpleNamespace(cfg={}, task_python_executable=str(root / 'bin/python'))
    assert task_python_executable(ctx) == root / 'bin/python'
