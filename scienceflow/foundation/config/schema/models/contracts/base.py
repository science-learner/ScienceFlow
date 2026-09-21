"""Shared configuration imports and immutable loader constants."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_CFG = Path(__file__).parents[3] / "default.yaml"
_CONFIG_INCLUDE_KEYS: tuple[str, ...] = ("include", "includes")

__all__ = tuple(name for name in globals() if not name.startswith("__"))
