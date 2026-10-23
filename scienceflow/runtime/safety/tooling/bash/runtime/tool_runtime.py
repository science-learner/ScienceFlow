"""Bash tool dispatch, deadlines, and hard-fuse policy."""

from __future__ import annotations

import time
from collections.abc import Callable

from inquirycraft.tools import ToolResult


def _apply_hard_fuse_timeout(self, timeout_sec: float) -> float:
    timeout = max(1.0, float(timeout_sec or 1.0))
    try:
        deadline = float(self.bash_hard_fuse_deadline_monotonic or 0.0)
    except (TypeError, ValueError):
        deadline = 0.0
    if deadline <= 0.0:
        return timeout
    try:
        reserve = max(0.0, float(self.bash_hard_fuse_finalization_reserve_sec or 0.0))
    except (TypeError, ValueError):
        reserve = 0.0
    _ = reserve
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return 1.0
    return max(1.0, min(timeout, remaining))


def _deadline_state(self) -> dict[str, float | bool]:
    try:
        deadline = float(self.bash_hard_fuse_deadline_monotonic or 0.0)
    except (TypeError, ValueError):
        deadline = 0.0
    try:
        reserve = max(0.0, float(self.bash_hard_fuse_finalization_reserve_sec or 0.0))
    except (TypeError, ValueError):
        reserve = 0.0
    if deadline <= 0.0:
        return {"deadline_event": False, "deadline_remaining_sec": 0.0, "finalization_reserve_sec": reserve}
    remaining = max(0.0, deadline - time.monotonic())
    return {
        "deadline_event": bool(reserve > 0.0 and remaining <= reserve),
        "deadline_remaining_sec": remaining,
        "finalization_reserve_sec": reserve,
    }


async def execute(
    self,
    command: str,
    *,
    on_output: Callable[[str], None] | None = None,
    **kwargs,
) -> ToolResult:
    """Run the explicit preflight/admission/execution/monitor/postprocess pipeline."""

    return await self._execute_with_stream_monitoring(
        command,
        on_output=on_output,
        **kwargs,
    )


async def _execute_with_stream_monitoring(
    self,
    command: str,
    *,
    on_output: Callable[[str], None] | None = None,
    **kwargs,
) -> ToolResult:
    """Compatibility entry point for callers using the legacy method name."""

    return await self._execute_process_pipeline(
        command,
        on_output=on_output,
        **kwargs,
    )
