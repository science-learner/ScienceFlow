"""User/project dotenv precedence and installed-package configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scienceflow.foundation.config.runtime.environment import (
    bootstrap_dotenv,
    user_env_path,
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in list(os.environ):
        if key.startswith(
            ("SCIENCEFLOW_", "CODE_", "FEEDBACK_", "XDG_", "API_", "BASE_URL")
        ) or key in {"MODEL", "MODELS"}:
            os.environ.pop(key)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


def write_env(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_user_config_is_loaded_without_source_checkout(tmp_path):
    path = write_env(user_env_path(), "CODE_MODELS=user-model\nCODE_API_KEYS=a,b\n")
    assert bootstrap_dotenv(repo_root=tmp_path / "site-packages") == (path,)
    assert os.environ["CODE_MODELS"] == "user-model"
    assert os.environ["CODE_API_KEYS"] == "a,b"


def test_shell_explicit_project_user_precedence(tmp_path, monkeypatch):
    write_env(user_env_path(), "CODE_MODELS=user\nUSER_ONLY=1\n")
    write_env(tmp_path / ".env", "CODE_MODELS=project\nPROJECT_ONLY=1\n")
    explicit = write_env(tmp_path / "selected.env", "CODE_MODELS=explicit\n")
    monkeypatch.setenv("CODE_MODELS", "shell")
    bootstrap_dotenv(env_file=explicit)
    assert os.environ["CODE_MODELS"] == "shell"
    assert os.environ["USER_ONLY"] == "1"
    assert os.environ["PROJECT_ONLY"] == "1"
    os.environ.pop("CODE_MODELS")
    bootstrap_dotenv(env_file=explicit)
    assert os.environ["CODE_MODELS"] == "explicit"
    os.environ.pop("CODE_MODELS")
    bootstrap_dotenv()
    assert os.environ["CODE_MODELS"] == "project"


def test_project_overrides_source_checkout_and_site_packages_is_ignored(
    tmp_path, monkeypatch
):
    repo = tmp_path / "source"
    (repo / "scienceflow").mkdir(parents=True)
    (repo / "pyproject.toml").touch()
    write_env(repo / ".env", "CODE_MODELS=source\n")
    project = tmp_path / "project"
    write_env(project / ".env", "CODE_MODELS=project\n")
    monkeypatch.chdir(project)
    bootstrap_dotenv(repo_root=repo)
    assert os.environ["CODE_MODELS"] == "project"
    os.environ.pop("CODE_MODELS")
    (project / ".env").unlink()
    site = tmp_path / "site-packages"
    write_env(site / ".env", "CODE_MODELS=unexpected\n")
    bootstrap_dotenv(repo_root=site)
    assert "CODE_MODELS" not in os.environ


def test_xdg_and_explicit_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert user_env_path() == tmp_path / "xdg/scienceflow/.env"
    explicit = write_env(tmp_path / "custom.env", "CODE_MODELS=custom\n")
    monkeypatch.setenv("SCIENCEFLOW_ENV_FILE", str(explicit))
    assert user_env_path() == explicit
    bootstrap_dotenv()
    assert os.environ["CODE_MODELS"] == "custom"


def test_legacy_home_config_and_explicit_missing_file(tmp_path, monkeypatch):
    write_env(Path.home() / ".scienceflow/.env", "CODE_MODELS=legacy\n")
    bootstrap_dotenv()
    assert os.environ["CODE_MODELS"] == "legacy"
    monkeypatch.setenv("SCIENCEFLOW_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(ValueError, match="does not exist"):
        bootstrap_dotenv()


def test_legacy_override_switch_keeps_file_priority(tmp_path, monkeypatch):
    write_env(user_env_path(), "CODE_MODELS=user\n")
    write_env(tmp_path / ".env", "CODE_MODELS=project\n")
    monkeypatch.setenv("CODE_MODELS", "shell")
    monkeypatch.setenv("SCIENCEFLOW_DOTENV_OVERRIDE", "1")
    bootstrap_dotenv()
    assert os.environ["CODE_MODELS"] == "project"


def test_exported_code_pool_does_not_change_feedback(monkeypatch):
    write_env(
        user_env_path(),
        "CODE_API_KEYS=user-code\nFEEDBACK_API_KEYS=user-feedback\nFEEDBACK_MODELS=judge\n",
    )
    monkeypatch.setenv("CODE_API_KEYS", "exported-a,exported-b")
    monkeypatch.setenv("CODE_MODELS", "code-model")
    bootstrap_dotenv()
    assert os.environ["CODE_API_KEYS"] == "exported-a,exported-b"
    assert os.environ["CODE_MODELS"] == "code-model"
    assert os.environ["FEEDBACK_API_KEYS"] == "user-feedback"
    assert os.environ["FEEDBACK_MODELS"] == "judge"


def test_unscoped_variables_do_not_populate_either_stage(monkeypatch):
    monkeypatch.setenv("API_KEYS", "shared-key")
    monkeypatch.setenv("BASE_URLS", "https://shared/v1")
    monkeypatch.setenv("MODELS", "shared-model")
    bootstrap_dotenv()
    for stage in ("CODE", "FEEDBACK"):
        for field in ("API_KEYS", "BASE_URLS", "MODELS"):
            assert f"{stage}_{field}" not in os.environ


def test_project_code_settings_leave_user_feedback_unchanged(tmp_path):
    write_env(
        user_env_path(), "CODE_API_KEYS=user-code\nFEEDBACK_API_KEYS=user-feedback\n"
    )
    write_env(tmp_path / ".env", "CODE_API_KEYS=project-code\n")
    bootstrap_dotenv()
    assert os.environ["CODE_API_KEYS"] == "project-code"
    assert os.environ["FEEDBACK_API_KEYS"] == "user-feedback"
