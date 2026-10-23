# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Top-level LNR lifecycle kernel."""

from __future__ import annotations

import asyncio
from typing import Any

from scienceflow.runtime.core.kernel.hooks import HookEvent, HookPoint, HookReport
from scienceflow.runtime.core.kernel.state_machines import (
    RunLifecycleEvent,
    RunLifecycleMachine,
)
from scienceflow.research.solver.lnr.orchestration.runtime.services.models import RunMode, RunSpec
from scienceflow.research.solver.lnr.orchestration.runtime.services.facade import RuntimeServices


class LnrRuntime:
    """Own lifecycle transitions, hooks, and single/multi runner selection."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.last_lifecycle: RunLifecycleMachine | None = None
        self.last_hook_reports: tuple[HookReport, ...] = ()

    async def run(self, spec: RunSpec | None = None) -> dict[str, Any]:
        effective = spec or self.services.spec_factory()
        run_id = effective.run_id or effective.worker_id or "lnr"
        lifecycle = RunLifecycleMachine(run_id=run_id)
        reports: list[HookReport] = []
        self.last_lifecycle = lifecycle

        async def emit(point: HookPoint, payload: dict[str, Any]) -> None:
            reports.append(
                await self.services.hooks.dispatch(
                    HookEvent(
                        point=point,
                        run_id=run_id,
                        worker_id=effective.worker_id,
                        event_id=f"{run_id}:{point.value}",
                        payload=payload,
                    )
                )
            )
            self.last_hook_reports = tuple(reports)

        mode_payload = {
            "mode": effective.mode.value,
            "worker_count": effective.worker_count,
        }
        await emit(HookPoint.RUN_STARTING, mode_payload)
        lifecycle.advance(RunLifecycleEvent.START, event_id=f"{run_id}:start")
        await emit(HookPoint.RUN_STARTED, mode_payload)

        runner = (
            self.services.run_single
            if effective.mode is RunMode.SINGLE_WORKER
            else self.services.run_multi
        )
        try:
            result = await runner()
        except asyncio.CancelledError:
            lifecycle.advance(RunLifecycleEvent.CANCEL, event_id=f"{run_id}:cancel")
            await emit(HookPoint.RUN_CANCELLED, mode_payload)
            raise
        except Exception as exc:
            lifecycle.advance(RunLifecycleEvent.FAIL, event_id=f"{run_id}:fail")
            await emit(
                HookPoint.RUN_FAILED,
                {**mode_payload, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        lifecycle.advance(RunLifecycleEvent.SUCCEED, event_id=f"{run_id}:succeed")
        await emit(HookPoint.RUN_FINISHED, mode_payload)
        return result


__all__ = ["LnrRuntime"]
