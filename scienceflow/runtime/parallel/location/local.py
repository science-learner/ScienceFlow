"""Allocate an exclusive task directory without moving user data."""

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


def resolve_local_root(management_workspace: Path, requested_root: str) -> Path:
    text = str(requested_root).strip()
    if not text or "\x00" in text:
        raise ValueError("Execution root must be a non-empty local directory.")
    if "://" in text or re.match(r"^[^/]+:", text):
        raise ValueError(
            "Remote execution is not supported yet; use a local execution root."
        )
    try:
        root = Path(text).expanduser()
        root = (root if root.is_absolute() else management_workspace / root).resolve()
    except RuntimeError as exc:
        raise ValueError(f"Cannot resolve local execution root: {text}") from exc
    if root.exists() and not root.is_dir():
        raise ValueError(f"Execution root is not a directory: {root}")
    return root


@dataclass(frozen=True, slots=True)
class LocalExecutionLocation:
    management_workspace: Path
    allocation_root: Path
    task_workspace: Path
    allocation_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "backend": "local",
            "management_workspace": str(self.management_workspace),
            "allocation_root": str(self.allocation_root),
            "task_workspace": str(self.task_workspace),
            "allocation_id": self.allocation_id,
        }


def allocate_local_workspace(
    management_workspace: Path,
    *,
    task_number: int | None = None,
    requested_root: str = "",
) -> LocalExecutionLocation:
    """Default numbered tasks keep their layout; explicit roots get unique children.

    Only a newly created, empty child may be removed by the caller on failure.
    A console task without an override retains its management workspace.
    """
    management = Path(management_workspace).expanduser().resolve()
    if task_number is not None and task_number < 1:
        raise ValueError("Task number must be positive.")
    allocation_id = uuid4().hex
    if requested_root:
        root = resolve_local_root(management, requested_root)
        prefix = f"task-{task_number:04d}" if task_number is not None else "task"
        target = root / f"{prefix}-{allocation_id}"
    elif task_number is not None:
        root = management / "tasks"
        target = root / f"task-{task_number:04d}"
        if not target.resolve().is_relative_to(management):
            raise ValueError(
                "Default task directory must remain inside the management workspace."
            )
    else:
        management.mkdir(parents=True, exist_ok=True)
        return LocalExecutionLocation(management, management, management, allocation_id)
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return LocalExecutionLocation(
        management, root.resolve(), target.resolve(), allocation_id
    )


def validate_record_location(record: dict) -> None:
    """Never resume into a missing, replaced or mismatched execution directory."""
    for root in record.get("task_roots", ()):
        path = Path(root)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError(
                f"Recorded task root has moved or contains a symlink: {path}"
            )
    location = record.get("execution_location") or {}
    if not location:
        return
    target = Path(location.get("task_workspace", ""))
    if location.get("backend") != "local" or not target.is_absolute():
        raise ValueError("Invalid local execution location in run record.")
    if not target.is_dir() or target.resolve() != target:
        raise ValueError(f"Recorded task workspace is missing or has moved: {target}")
    if str(target) != record.get("control_workspace"):
        raise ValueError("Execution location does not match the task workspace.")
    if any(
        not Path(root).resolve().is_relative_to(target)
        for root in record.get("task_roots", ())
    ):
        raise ValueError(
            "Manifest task roots must remain inside the allocated workspace."
        )
