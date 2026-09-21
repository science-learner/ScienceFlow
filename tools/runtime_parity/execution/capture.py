from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.runtime_parity.comparison.normalize import normalize_value
from .workspace import capture_workspace


def capture_case(
    case_id: str,
    context: Any,
    mechanism: Any,
    workspace: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "context": normalize_value(context, workspace=workspace),
        "mechanism": normalize_value(mechanism, workspace=workspace),
        "workspace": capture_workspace(workspace),
    }


__all__ = ["capture_case"]
