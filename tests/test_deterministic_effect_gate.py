from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scienceflow.foundation.config.llm import llm_factory as agent_runtime
from scienceflow.foundation.config.schema.settings import StageConfig
from scienceflow.research.solver.lnr.orchestration.solver import (
    LnrSolver,
    _deterministic_gate_enabled,
    _deterministic_gate_timestamp,
)
from scienceflow.research.state.workspace.storage.git import ensure_workspace_source_git
from scienceflow.runtime.safety.tooling.bash import BashTool


def _head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def test_fixed_commit_timestamp_produces_identical_initial_workspace_sha(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for workspace in (first, second):
        workspace.mkdir()
        (workspace / "solver.py").write_text("VALUE = 1\n", encoding="utf-8")
        result = ensure_workspace_source_git(
            workspace,
            track_globs=["*.py"],
            commit_timestamp="2000-01-01T00:00:00Z",
        )
        assert result.ready and result.committed

    assert _head(first) == _head(second)


def test_gate_environment_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIENCEFLOW_DETERMINISTIC_GATE", raising=False)
    monkeypatch.delenv("SCIENCEFLOW_DETERMINISTIC_GATE_TIMESTAMP_UTC", raising=False)
    assert _deterministic_gate_enabled() is False
    assert _deterministic_gate_timestamp() is None

    monkeypatch.setenv("SCIENCEFLOW_DETERMINISTIC_GATE", "1")
    assert _deterministic_gate_enabled() is True
    assert _deterministic_gate_timestamp() == "2000-01-01T00:00:00Z"


def test_gate_stage_override_uses_lnr_seed_without_changing_normal_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = StageConfig(model="local-model", api_key="key", base_url="http://localhost/v1")
    fake = SimpleNamespace(
        worker_count=2,
        worker_index=1,
        worker_id="W01",
        lhr=SimpleNamespace(seed=2222),
        cfg=SimpleNamespace(agent=SimpleNamespace(code=stage), exp_id="circle-packing"),
    )
    fake._worker_uid_prefix = lambda: "W01"

    monkeypatch.delenv("SCIENCEFLOW_DETERMINISTIC_GATE", raising=False)
    assert LnrSolver._worker_llm_stage_override(fake, "code") is None

    monkeypatch.setenv("SCIENCEFLOW_DETERMINISTIC_GATE", "true")
    override = LnrSolver._worker_llm_stage_override(fake, "code")
    assert override is not stage
    assert override.request_seed == 2223
    assert not hasattr(stage, "request_seed")


def test_gate_freezes_provider_visible_runtime_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        deadline=time.monotonic() + 17,
        lhr=SimpleNamespace(wall_clock_budget_sec=900),
    )
    agent = SimpleNamespace(_bash_timeout_sec=14400)

    monkeypatch.setenv("SCIENCEFLOW_DETERMINISTIC_GATE", "1")
    context = LnrSolver._runtime_context_for_agent(fake, agent)

    assert "wall_clock_remaining_sec: 900" in context
    assert "effective_bash_timeout_sec: 900" in context


def test_gate_exposes_logical_not_physical_cpu_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        worker_index=1,
        worker_extra_env={
            "SCIENCEFLOW_WORKER_CPU_LIST": "64-71",
            "SCIENCEFLOW_TASK_CPU_LIST": "56-71",
        },
        cfg=SimpleNamespace(exec=SimpleNamespace(cpu_list="56-71", gpu_list="cpu")),
    )
    fake._worker_uid_prefix = lambda: "W01"

    monkeypatch.delenv("SCIENCEFLOW_DETERMINISTIC_GATE", raising=False)
    normal = LnrSolver._allocated_compute_context_lines(fake)
    assert "  - worker_cpu_list: 64-71" in normal
    assert "  - task_cpu_list: 56-71" in normal

    monkeypatch.setenv("SCIENCEFLOW_DETERMINISTIC_GATE", "1")
    gated = LnrSolver._allocated_compute_context_lines(fake)
    assert "  - worker_cpu_list: 8-15" in gated
    assert "  - task_cpu_list: 0-15" in gated
    assert all("56-71" not in line and "64-71" not in line for line in gated)


def test_build_llm_only_forwards_dynamically_attached_request_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class CapturingPool:
        @classmethod
        def from_endpoints(cls, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace()

    monkeypatch.setattr(agent_runtime, "PooledLLM", CapturingPool)
    stage = StageConfig(model="local-model", api_key="key", base_url="http://localhost/v1")

    agent_runtime.build_stage_llm(stage)
    assert "request_seed" not in captured[-1]

    stage.request_seed = 3333
    agent_runtime.build_stage_llm(stage)
    assert captured[-1]["request_seed"] == 3333


@pytest.mark.asyncio
async def test_gate_hides_only_dynamic_bash_header_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )
    monkeypatch.delenv("SCIENCEFLOW_DETERMINISTIC_GATE", raising=False)
    normal = await tool.execute("echo ok")
    assert normal.output.startswith("[exit=0, ")
    assert normal.output.rstrip().endswith("ok")

    monkeypatch.setenv("SCIENCEFLOW_DETERMINISTIC_GATE", "1")
    gated = await tool.execute("echo ok")
    assert gated.output == "[exit=0]\nok"
