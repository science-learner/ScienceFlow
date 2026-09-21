"""Shared imports and immutable parallel-runner values."""

from __future__ import annotations
import logging
import sys
from typing import Any
from scienceflow.runtime.environment import resolve_project_python as _resolve_project_python_impl
logger = logging.getLogger("scienceflow")


def _public_override(name: str, default: Any) -> Any:
    public_module = sys.modules.get("scienceflow.runtime.parallel.execution.runner")
    candidate = getattr(public_module, name, None)
    return candidate if candidate is not None and candidate is not default else None


def resolve_project_python() -> tuple[str, str]:
    override = _public_override("resolve_project_python", resolve_project_python)
    if override is not None:
        return override()
    return _resolve_project_python_impl()

__all__ = tuple(name for name in globals() if not name.startswith("__"))
