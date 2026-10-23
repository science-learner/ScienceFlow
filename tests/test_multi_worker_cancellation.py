"""A failed/cancelled coordinator must await every worker's cleanup hook."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scienceflow.research.solver.lnr.orchestration.runtime.execution.multi_worker import (
    run_multi_worker,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_coordinator_waits_for_worker_cleanup_before_closing_owner(
    tmp_path, cancel
):
    entered, cleaned, aggregator_cleaned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    order = []

    async def worker(request):
        if request.worker_index == 0:
            await entered.wait()
            if not cancel:
                raise RuntimeError("worker failed")
            await asyncio.sleep(60)
        try:
            entered.set()
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(0.01)
            order.append("worker cleaned")
            cleaned.set()

    async def aggregate(*args):
        try:
            await asyncio.sleep(60)
        finally:
            aggregator_cleaned.set()

    async def close_owner():
        assert cleaned.is_set()
        order.append("owner closed")

    services = SimpleNamespace(
        spec=SimpleNamespace(
            worker_count=2,
            final_artifact_mode="workspace",
            merge_enabled=True,
            root_dir=tmp_path,
            log_dir=tmp_path,
        ),
        status_projection=SimpleNamespace(write=Mock()),
        append_event=Mock(),
        run_one_worker=worker,
        aggregate_worker_state_periodically=aggregate,
        close_merge_owner_agent=close_owner,
    )
    task = asyncio.create_task(run_multi_worker(services))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert order == ["worker cleaned", "owner closed"]
    assert aggregator_cleaned.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [True, False])
async def test_aggregation_exit_immediately_reaps_workers(tmp_path, failure):
    entered = asyncio.Event()
    cleaned = []

    async def worker(request):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.append(request.worker_index)

    async def aggregate(*args):
        await entered.wait()
        if failure:
            raise ValueError("observation integrity failure")

    async def close_owner():
        assert sorted(cleaned) == [0, 1]

    services = SimpleNamespace(
        spec=SimpleNamespace(
            worker_count=2,
            final_artifact_mode="workspace",
            merge_enabled=True,
            root_dir=tmp_path,
            log_dir=tmp_path,
        ),
        status_projection=SimpleNamespace(write=Mock()),
        append_event=Mock(),
        run_one_worker=worker,
        aggregate_worker_state_periodically=aggregate,
        close_merge_owner_agent=close_owner,
    )
    with pytest.raises(
        ValueError if failure else RuntimeError,
        match="observation integrity failure" if failure else "stopped unexpectedly",
    ):
        await asyncio.wait_for(run_multi_worker(services), timeout=1)
    assert sorted(cleaned) == [0, 1]
