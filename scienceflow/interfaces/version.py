"""Report the version of the ScienceFlow code being executed."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version() -> str:
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.is_file():
        project = tomllib.loads(project_file.read_text(encoding="utf-8")).get(
            "project", {}
        )
        if project.get("name", "").casefold() == "scienceflow":
            return str(project["version"])
    try:
        return version("scienceflow")
    except PackageNotFoundError:
        return "dev"
