"""Structural contracts for layered Bash tooling and monitor owners."""

from __future__ import annotations

from pathlib import Path

from scienceflow.runtime.safety.tooling.bash.monitor.execution import (
    BashExecutionMonitor,
    run_monitored_execution,
)


def test_bash_tooling_uses_bounded_owner_layers() -> None:
    root = Path(__file__).parents[1] / "scienceflow/runtime/safety/tooling/bash"
    direct_modules = {
        path.name for path in root.glob("*.py") if path.name != "__init__.py"
    }
    assert direct_modules == set()
    assert {
        path.name for path in root.iterdir() if path.is_dir() and path.name != "__pycache__"
    } == {"core", "monitor", "policy", "process", "runtime"}
    assert {
        path.name for path in (root / "core").glob("*.py") if path.name != "__init__.py"
    } == {
        "pipeline_preparation.py",
        "pure.py",
        "request_builder.py",
        "shared.py",
        "tool.py",
    }
    assert {
        path.name for path in (root / "runtime").glob("*.py") if path.name != "__init__.py"
    } == {
        "tool_runtime.py",
    }

    limits = {
        "monitor/effects.py": 250,
        "monitor/execution.py": 200,
        "monitor/guard.py": 600,
        "monitor/stream.py": 400,
        "policy/admission.py": 800,
        "policy/output.py": 400,
        "process/pipeline.py": 200,
        "process/start.py": 200,
        "process/streaming.py": 300,
    }
    observed = {
        name: len((root / name).read_text(encoding="utf-8").splitlines())
        for name in limits
    }
    assert all(observed[name] <= limit for name, limit in limits.items()), observed


def test_bash_monitor_composes_real_guard_and_effect_owners() -> None:
    assert run_monitored_execution.__module__ == (
        "scienceflow.runtime.safety.tooling.bash.monitor.execution"
    )
    assert [owner.__module__ for owner in BashExecutionMonitor.__mro__[1:3]] == [
        "scienceflow.runtime.safety.tooling.bash.monitor.guard",
        "scienceflow.runtime.safety.tooling.bash.monitor.effects",
    ]
