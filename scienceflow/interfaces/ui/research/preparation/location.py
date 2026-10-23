"""Resolve placement before preparation tools can write any task artifacts."""

import asyncio
import json
import re
import shlex
from pathlib import Path

from scienceflow.runtime.parallel.location import (
    LocalExecutionLocation,
    allocate_local_workspace,
    resolve_local_root,
)

from .runtime import run_preparation_task

_LOCATION_INTENT = re.compile(
    r"workspace|execution.root|工作目录|工作区|执行目录|输出目录|放到|放在|保存到|"
    r"(?:run|execute|save|place|output)\b.{0,40}\b(?:in|under|to|at)\s+(?:[/~.]|[\"'])",
    re.IGNORECASE,
)
_NAMED_LOCATION_INTENT = re.compile(
    r"\bworkspace\b|execution.root|工作目录|工作区|执行目录|输出目录|"
    r"(?:放到|放在|保存到).{0,20}(?:目录|路径|磁盘)",
    re.IGNORECASE,
)


def extract_execution_root(text: str) -> tuple[str, str]:
    """Only consume explicit placement options; leave other task syntax alone."""
    if not re.search(r"(?:^|\s)(?:workspace|execution_root|execution-root)=", text):
        return text, ""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"Invalid workspace option: {exc}") from exc
    roots, remaining = [], []
    for token in tokens:
        key, separator, value = token.partition("=")
        if separator and key in {"workspace", "execution_root", "execution-root"}:
            if not value.strip():
                raise ValueError("workspace= requires a local directory.")
            roots.append(value)
        else:
            remaining.append(token)
    if len(roots) != 1:
        raise ValueError("Specify one execution root per task.")
    return shlex.join(remaining), roots[0]


def needs_location_resolution(constraints: str) -> bool:
    _, root = extract_execution_root(constraints)
    has_path = re.search(
        r"(?:^|[\s\"'=]|到|在|录)(?:/|~/|\.\.?/)[^\s，;；]+", constraints
    )
    return not root and bool(
        _NAMED_LOCATION_INTENT.search(constraints)
        or (has_path and _LOCATION_INTENT.search(constraints))
    )


async def resolve_natural_location(owner, host, constraints: str) -> str:
    prompt = (
        "Extract only the explicitly requested research execution directory from the user text. "
        "Do not infer a location from dataset/evaluator paths. Do not run tools. "
        'Return JSON {"execution_root": null} if none was requested, otherwise '
        '{"execution_root": "exact directory as written by the user"}. '
        'If a location was requested but is ambiguous or not spelled out, return {"execution_root": ""}. '
        "Never invent, expand, or normalize the path.\nUser text: " + constraints
    )
    try:
        response = await run_preparation_task(owner, host, prompt, tool_free=True)
        payload = json.loads(response)
        if not isinstance(payload, dict) or set(payload) != {"execution_root"}:
            raise ValueError("Invalid location response.")
        root = payload["execution_root"]
        if root is not None and (
            not isinstance(root, str) or not root.strip() or root not in constraints
        ):
            raise ValueError("Location extraction did not match the user's directory.")
        if root:
            resolve_local_root(owner.workspace, root)
        return root or ""
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Cannot determine execution location; use workspace=/local/directory."
        ) from exc


async def allocate_preparation_location(
    owner, host, constraints, *, requested_root=None
):
    constraints, root = extract_execution_root(constraints)
    if requested_root is not None:
        root = requested_root
    if root:
        root = str(resolve_local_root(owner.workspace, root))
    if callable(getattr(owner._managed, "new_workspace", None)):
        await owner._managed.new_workspace(constraints, host, execution_root=root)
        record = owner._managed.preparing["execution_location"]
        location = LocalExecutionLocation(
            Path(record["management_workspace"]),
            Path(record["allocation_root"]),
            Path(record["task_workspace"]),
            record["allocation_id"],
        )
    else:
        location = await asyncio.to_thread(
            allocate_local_workspace, owner.workspace, requested_root=root
        )
    return constraints, location
