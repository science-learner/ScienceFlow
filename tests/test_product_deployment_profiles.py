from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        assert match, requirement
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_default_install_is_light_cpu_only_control_plane() -> None:
    metadata = _project_metadata()
    core = _requirement_names(metadata["project"]["dependencies"])
    forbidden = {
        "catboost",
        "lightgbm",
        "mlebench",
        "pandas",
        "pytest",
        "scikit-learn",
        "tabpfn",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "ultralytics",
        "xgboost",
    }
    assert not core & forbidden
    assert {"httpx", "inquirycraft", "openai"} <= core
    assert metadata["tool"]["uv"]["default-groups"] == []
    assert _requirement_names(metadata["dependency-groups"]["dev"]) == {
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
    }


def test_full_profile_covers_task_profiles() -> None:
    extras = _project_metadata()["project"]["optional-dependencies"]
    assert {"ml", "gpu", "mlebench", "detection-2d", "full"} <= set(extras)
    full = _requirement_names(extras["full"])
    task_profiles = set().union(
        *(
            _requirement_names(extras[name])
            for name in ("ml", "gpu", "mlebench", "detection-2d")
        )
    )
    assert task_profiles <= full
    assert "torch" not in _requirement_names(extras["ml"])
    assert "tabpfn" not in _requirement_names(extras["ml"])


def test_public_install_surface_is_light_and_full() -> None:
    public_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "docs" / "public" / "README_CN.md")
    )
    assert "pip install scienceflow" in public_docs
    assert 'pip install "scienceflow[full]"' in public_docs
    for internal_extra in ("ml", "gpu", "mlebench"):
        assert f'scienceflow[{internal_extra}]' not in public_docs


def test_requirements_txt_matches_core_profile() -> None:
    metadata = _project_metadata()
    core = _requirement_names(metadata["project"]["dependencies"])
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert _requirement_names(lines) == core


def test_deployment_targets_share_one_source_tree() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    for target in ("core", "ml", "gpu", "mlebench", "full"):
        assert f" AS {target}\n" in dockerfile
    assert 'ENTRYPOINT ["scienceflow"]' in dockerfile
    assert "COPY scienceflow ./scienceflow" in dockerfile

    compose = (ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    assert "target: ${SCIENCEFLOW_TARGET:-core}" in compose
    assert "SCIENCEFLOW_MODEL_CONFIG" in compose
    assert "/home/scienceflow/.config/scienceflow/models.json" in compose
    assert "  vllm:" not in compose
