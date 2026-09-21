"""Static contracts for the V5 formal and product logic acceptance manifests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = (
    ROOT
    / "scienceflow"
    / "foundation"
    / "config"
    / "profiles"
    / "acceptance"
    / "v5"
)


def test_v5_agent_runtime_ownership_is_physically_enforced() -> None:
    package_root = ROOT / "scienceflow"
    agent_root = package_root / "agent"
    files = sorted(agent_root.rglob("*.py"))
    direct_business_files = [
        path for path in agent_root.glob("*.py") if path.name != "__init__.py"
    ]
    line_counts = {
        path: len(path.read_text(encoding="utf-8").splitlines()) for path in files
    }
    # V5.2 permits nested owner packages while bounding the root, each owner,
    # and aggregate growth. The former recursive file-count cap punished the
    # requested multi-level layout even when responsibilities became narrower.
    assert len(direct_business_files) <= 8
    assert max(line_counts.values()) <= 800
    assert sum(line_counts.values()) <= 4_336
    assert not (package_root / "runtime" / "orchestrator.py").exists()
    for retired in ("components", "context", "llm", "memory", "run_control", "runtime"):
        assert not list((agent_root / retired).rglob("*.py"))

    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for retired_symbol in (
        "HostedAgentRuntime",
        "LLMStreamMixin",
        "RunLoopMixin",
        "SingleToolExecMixin",
        "_run_scienceflow_policy_loop",
        "runtime_backend",
    ):
        assert retired_symbol not in source

    tree = ast.parse(
        (agent_root / "core" / "runtime" / "agent.py").read_text(encoding="utf-8")
    )
    science_agent = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScienceAgent"
    )
    assert [base.id for base in science_agent.bases if isinstance(base, ast.Name)] == [
        "BaseModel"
    ]


@pytest.mark.parametrize(
    ("name", "exp_id", "input_data_dir", "cpu_list", "omp_threads", "config"),
    (
        (
            "circle_packing_w2_1h.yaml",
            "circle-packing",
            "/home/mingming/miing_data/opt_solver/circle_packing",
            "0-15",
            8,
            "scienceflow/foundation/config/default.yaml",
        ),
        (
            "nomad2018_deep_w2_1h.yaml",
            "nomad2018-predict-transparent-conductors",
            "/work/miing_data/mlebench_all_data/nomad2018-predict-transparent-conductors/prepared/dataset_split/Deep",
            "16-31",
            8,
            "scienceflow/foundation/config/default.yaml",
        ),
        (
            "tfbind8_w2_1h.yaml",
            "sci-modeling-bench-tfbind8",
            "/home/mingming/miing_data/sci_modeling_bench/tfbind8/public",
            "0-15",
            4,
            "scienceflow/foundation/config/sci_modeling_bench.yaml",
        ),
    ),
)
def test_v5_acceptance_manifest_is_exactly_w2_one_hour(
    name: str,
    exp_id: str,
    input_data_dir: str,
    cpu_list: str,
    omp_threads: int,
    config: str,
) -> None:
    manifest = yaml.safe_load((MANIFEST_ROOT / name).read_text(encoding="utf-8"))
    assert manifest["max_concurrent"] == 1
    assert manifest["time_limit"] == 4200
    assert manifest["resume"] is False
    assert manifest["defaults"]["config"] == config
    assert manifest["defaults"]["workspace_base"] == "./workspaces/v5_acceptance"
    assert len(manifest["tasks"]) == 1

    task = manifest["tasks"][0]
    assert task["exp_id"] == exp_id
    assert task["run_id"].endswith("-w2-1h")
    assert task["input_data_dir"] == input_data_dir
    assert task["time_limit"] == 4200
    assert task["cpu_list"] == cpu_list
    assert task["gpu_list"] == "cpu"
    assert task["lnr"] == {
        "wall_clock_budget_sec": 3600,
        "num_workers": 2,
        "omp_threads_cap": omp_threads,
        "resource_gpu_pool": [],
        "seed": 2222,
    }


@pytest.mark.parametrize(
    ("name", "run_id", "input_data_dir", "cpu_list", "omp_threads", "config"),
    (
        (
            "circle_packing_w2_2h.yaml",
            "product-circle-packing-w2-2h-20260903-r1",
            "/home/mingming/miing_data/opt_solver/circle_packing",
            "0-15",
            8,
            "scienceflow/foundation/config/default.yaml",
        ),
        (
            "nomad2018_deep_w2_2h.yaml",
            "product-nomad2018-deep-w2-2h-20260903-r1",
            "/work/miing_data/mlebench_all_data/nomad2018-predict-transparent-conductors/prepared/dataset_split/Deep",
            "16-31",
            8,
            "scienceflow/foundation/config/default.yaml",
        ),
        (
            "tfbind8_w2_2h.yaml",
            "product-tfbind8-w2-2h-20260903-r1",
            "/home/mingming/rsi/codes/ScienceFlow_Noah_Beta/cache/sci_modeling_bench/tfbind8/public",
            "32-47",
            4,
            "scienceflow/foundation/config/sci_modeling_bench.yaml",
        ),
    ),
)
def test_product_logic_acceptance_manifest_is_exactly_w2_two_hours(
    name: str,
    run_id: str,
    input_data_dir: str,
    cpu_list: str,
    omp_threads: int,
    config: str,
) -> None:
    manifest = yaml.safe_load((MANIFEST_ROOT / name).read_text(encoding="utf-8"))
    assert manifest["max_concurrent"] == 1
    assert manifest["time_limit"] == 7800
    assert manifest["resume"] is False
    assert manifest["defaults"]["config"] == config
    assert manifest["defaults"]["workspace_base"] == (
        "./workspaces/product_2h_acceptance_20260903_r1"
    )
    assert len(manifest["tasks"]) == 1

    task = manifest["tasks"][0]
    assert task["run_id"] == run_id
    assert task["input_data_dir"] == input_data_dir
    assert Path(input_data_dir).is_dir()
    assert task["time_limit"] == 7800
    assert task["cpu_list"] == cpu_list
    assert task["gpu_list"] == "cpu"
    assert task["lnr"] == {
        "wall_clock_budget_sec": 7200,
        "num_workers": 2,
        "omp_threads_cap": omp_threads,
        "resource_gpu_pool": [],
        "seed": 2222,
    }


def test_product_two_hour_cpu_domains_are_disjoint() -> None:
    names = (
        "circle_packing_w2_2h.yaml",
        "nomad2018_deep_w2_2h.yaml",
        "tfbind8_w2_2h.yaml",
    )
    cpu_sets: list[set[int]] = []
    for name in names:
        manifest = yaml.safe_load((MANIFEST_ROOT / name).read_text(encoding="utf-8"))
        start, end = map(int, manifest["tasks"][0]["cpu_list"].split("-", 1))
        cpu_sets.append(set(range(start, end + 1)))

    assert all(cpu_sets[left].isdisjoint(cpu_sets[right]) for left in range(3) for right in range(left + 1, 3))
