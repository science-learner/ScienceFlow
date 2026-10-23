# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Pytest hooks: route tmp workspaces under ``/tmp`` so tests do not write under the repo.

- ``tmp_path`` / ``tmp_path_factory`` create directories under :data:`tests.fs_root.TEST_WORKSPACE_ROOT`
  (default ``/tmp/scienceflow_tests``); each test’s directory is removed in ``finally``.
- After the session, :func:`pytest_sessionfinish` deletes only this process’s owned
  session directory; concurrent sessions are left intact. Set ``SCIENCEFLOW_TEST_KEEP_TMP=1`` to skip session cleanup for debugging.

**Repo root artifacts** (``logs/``, ``workspace/``, ``resolved_config.yaml`` under the ScienceFlow
package root): pytest does **not** create these in the repo; they come from CLI runs with default
``task_workspace_root_dir: "."`` or old files. This module does not remove them unless you set
``SCIENCEFLOW_TEST_CLEAN_REPO_ROOT=1`` (see :func:`pytest_sessionfinish`), which best-effort deletes
those paths after the session—use only when you accept losing local CLI outputs in the repo root.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from tests.fs_root import TEST_WORKSPACE_ROOT

# ScienceFlow repo root (parent of this ``tests/`` directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config) -> None:
    """Isolate defaults before collection imports and constructs CLI commands."""
    patch = pytest.MonkeyPatch()
    patch.setenv("XDG_CONFIG_HOME", str(TEST_WORKSPACE_ROOT / "collection-config"))
    config.add_cleanup(patch.undo)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_workspace_root() -> None:
    TEST_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def research_model_registry(isolated_config_home):
    """Local test data only; provider execution must be mocked by the caller."""
    path = isolated_config_home / "scienceflow" / "models.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "models": {
                "test-model": {
                    "model": "test-model",
                    "endpoints": [{"url": "http://127.0.0.1:1/v1", "key": "test-only"}],
                },
            },
            "defaults": {
                "code_models": ["test-model"],
                "feedback_models": ["test-model"],
            },
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def isolated_config_home(tmp_path_factory, monkeypatch):
    """Tests never inherit the developer's model registry or credentials."""
    root = tmp_path_factory.mktemp("user-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    return root


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Per-test directory under ``/tmp/scienceflow_tests`` (replaces pytest default basetemp)."""
    safe = request.node.name.replace("/", "_").replace("[", "_").replace("]", "_")[:80]
    d = Path(
        tempfile.mkdtemp(prefix=f"pytest_{safe}_", dir=str(TEST_WORKSPACE_ROOT)),
    )
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_path_factory(request: pytest.FixtureRequest):
    """Factory for extra dirs under ``/tmp/scienceflow_tests`` (pytest-compatible ``mktemp`` API)."""
    created: list[Path] = []

    class _Factory:
        def mktemp(self, basename: str) -> Path:
            p = Path(
                tempfile.mkdtemp(prefix=f"{basename}_", dir=str(TEST_WORKSPACE_ROOT)),
            )
            created.append(p)
            return p

    def _cleanup() -> None:
        for p in created:
            shutil.rmtree(p, ignore_errors=True)

    request.addfinalizer(_cleanup)
    return _Factory()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean this test session only; other processes own separate directories."""
    if os.environ.get("SCIENCEFLOW_TEST_KEEP_TMP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pass  # skip /tmp cleanup only; still allow opt-in repo-root cleanup below
    else:
        shutil.rmtree(TEST_WORKSPACE_ROOT, ignore_errors=True)

    _clean_repo_root_artifacts_if_requested()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _clean_repo_root_artifacts_if_requested() -> None:
    """Remove ScienceFlow root ``logs/``, ``workspace/``, ``resolved_config.yaml`` if env is set."""
    if not _env_truthy("SCIENCEFLOW_TEST_CLEAN_REPO_ROOT"):
        return
    root = _REPO_ROOT
    f = root / "resolved_config.yaml"
    if f.is_file():
        try:
            f.unlink()
        except OSError:
            pass
    for dirname in ("logs", "workspace"):
        d = root / dirname
        if d.is_dir():
            try:
                shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
