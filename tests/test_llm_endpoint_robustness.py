# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Transient LLM-endpoint robustness: same-round stream retries and pool failover.

Covers the retry classification / retry_after extraction added to
``scienceflow.core.agent.runtime.run_loop`` and ``scienceflow.core.key_pool``,
plus the worker-error reclassification in ``scienceflow.solver.lnr.solver``.
Motivation: a shared-upstream Cloudflare-tunnel 524 (retry_after only in the
JSON body) used to kill a long agent run outright; it must instead be retried
with a bounded backoff that honors the endpoint-asked delay.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from deepcraft_core import Memory

from scienceflow.core.agent import ScienceAgent
from scienceflow.core.agent.runtime.run_loop import (
    _LLM_RETRY_DELAY_CAP_SEC,
    _is_retryable_llm_error,
    _retry_after_seconds_from_exc,
)
from scienceflow.core import key_pool
from scienceflow.core.key_pool import (
    COOLDOWN_SERVER_ERROR,
    PooledLLM,
    _is_retryable_5xx_error,
    _retry_after_seconds_full,
)
from scienceflow.solver.lnr.solver import LnrSolver


def _http_response(status: int, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    return httpx.Response(status, request=request, json=json_body or {})


def _status_error(status: int, body: dict | None = None) -> openai.APIStatusError:
    resp = _http_response(status, body)
    return openai.InternalServerError("boom", response=resp, body=body) if status >= 500 else (
        openai.BadRequestError("bad", response=resp, body=body)
    )


# ---------------------------------------------------------------------------
# run_loop._is_retryable_llm_error
# ---------------------------------------------------------------------------


def test_retryable_llm_error_covers_5xx_connection_and_timeout() -> None:
    assert _is_retryable_llm_error(_status_error(524))
    assert _is_retryable_llm_error(_status_error(502))
    assert _is_retryable_llm_error(_status_error(503))
    assert _is_retryable_llm_error(_status_error(529))
    assert _is_retryable_llm_error(
        openai.APIConnectionError(request=httpx.Request("POST", "https://x/v1"))
    )
    assert _is_retryable_llm_error(httpx.ReadTimeout("deadline"))
    assert _is_retryable_llm_error(httpx.ConnectTimeout("connect deadline"))


def test_retryable_llm_error_keeps_empty_stream_classes() -> None:
    assert _is_retryable_llm_error(
        ValueError("Empty response from streaming tool LLM")
    )
    assert _is_retryable_llm_error(
        ValueError("Incomplete streaming tool call(s) from LLM; finish_reason=length")
    )
    # A non-timeout transport error is retryable; a timeout is handled above too.
    assert _is_retryable_llm_error(httpx.ReadError("stream disconnected"))


def test_retryable_llm_error_rejects_client_errors_and_value_errors() -> None:
    # 4xx (except the empty-stream ValueError phrasing) must NOT be retried here.
    assert not _is_retryable_llm_error(_status_error(400))
    assert not _is_retryable_llm_error(_status_error(401))
    assert not _is_retryable_llm_error(_status_error(404))
    assert not _is_retryable_llm_error(_status_error(429))  # rate limit: pool's job
    assert not _is_retryable_llm_error(ValueError("some unrelated failure"))
    assert not _is_retryable_llm_error(RuntimeError("plain runtime error"))


# ---------------------------------------------------------------------------
# run_loop._retry_after_seconds_from_exc
# ---------------------------------------------------------------------------


def test_retry_after_from_exc_reads_sdk_attribute() -> None:
    exc = _status_error(503)
    exc.retry_after = 42  # type: ignore[attr-defined]
    assert _retry_after_seconds_from_exc(exc) == 42.0


def test_retry_after_from_exc_reads_json_body() -> None:
    # The run3 Cloudflare-tunnel 524 case: retry_after only in the body.
    exc = _status_error(524, body={"retryable": True, "retry_after": 120})
    assert _retry_after_seconds_from_exc(exc) == 120.0


def test_retry_after_from_exc_reads_response_json_when_body_missing() -> None:
    resp = _http_response(524, {"retry-after": 90})
    exc = openai.InternalServerError("boom", response=resp, body=None)
    assert _retry_after_seconds_from_exc(exc) == 90.0


def test_retry_after_from_exc_returns_none_when_absent_or_invalid() -> None:
    assert _retry_after_seconds_from_exc(_status_error(503)) is None
    exc = _status_error(503, body={"retry_after": "not-a-number"})
    assert _retry_after_seconds_from_exc(exc) is None
    # Negative values are rejected (callers fall back to exponential backoff).
    exc_neg = _status_error(503, body={"retry_after": -5})
    assert _retry_after_seconds_from_exc(exc_neg) is None


# ---------------------------------------------------------------------------
# run_loop: same-round retry honors retry_after, capped
# ---------------------------------------------------------------------------


async def _sleep_recorder(record: list[float]):
    async def _sleep(delay: float) -> None:
        record.append(float(delay))

    return _sleep


@pytest.mark.asyncio
async def test_stream_retry_honors_retry_after_from_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 524 carrying retry_after=3 in its body drives a 3s same-round backoff."""
    delays: list[float] = []

    async def _recording_sleep(delay: float) -> None:
        delays.append(float(delay))

    monkeypatch.setattr(asyncio, "sleep", _recording_sleep)

    class _Gateway524LLM:
        model = "fake-524"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            if self.calls == 1:
                raise _status_error(524, body={"retryable": True, "retry_after": 3})
            return SimpleNamespace(tool_calls=[], content="recovered-after-524")

    llm = _Gateway524LLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=4,
        llm_tool_stream_retry_base_delay_sec=1.0,
        llm_tool_stream_retry_max_delay_sec=60.0,
    )
    out = await agent.run("hi")

    assert llm.calls == 2
    assert "recovered-after-524" in out
    # The endpoint-asked 3s wins over the 1s exponential base for attempt 0.
    assert 3.0 in delays


@pytest.mark.asyncio
async def test_stream_retry_caps_retry_after_at_300s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absurd retry_after (e.g. 9999) is capped at _LLM_RETRY_DELAY_CAP_SEC."""
    delays: list[float] = []

    async def _recording_sleep(delay: float) -> None:
        delays.append(float(delay))

    monkeypatch.setattr(asyncio, "sleep", _recording_sleep)

    class _GreedyRetryLLM:
        model = "fake-greedy"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            if self.calls == 1:
                raise _status_error(503, body={"retry_after": 9999})
            return SimpleNamespace(tool_calls=[], content="ok")

    llm = _GreedyRetryLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=3,
        llm_tool_stream_retry_base_delay_sec=1.0,
        llm_tool_stream_retry_max_delay_sec=60.0,
    )
    await agent.run("hi")

    assert llm.calls == 2
    assert max(delays) == _LLM_RETRY_DELAY_CAP_SEC == 300.0


@pytest.mark.asyncio
async def test_stream_retry_falls_back_to_exponential_without_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retry_after anywhere -> exponential backoff (base * 2**attempt)."""
    delays: list[float] = []

    async def _recording_sleep(delay: float) -> None:
        delays.append(float(delay))

    monkeypatch.setattr(asyncio, "sleep", _recording_sleep)

    class _ConnResetLLM:
        model = "fake-conn"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2

        def __init__(self) -> None:
            self.calls = 0

        async def ask_tool_stream(self, **kwargs: object) -> object:
            self.calls += 1
            handle = kwargs.get("handle")
            if handle is not None and not getattr(handle, "interrupted", False):
                handle.finish()
            if self.calls <= 2:
                raise openai.APIConnectionError(
                    request=httpx.Request("POST", "https://x/v1")
                )
            return SimpleNamespace(tool_calls=[], content="ok")

    llm = _ConnResetLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        llm_tool_stream_max_attempts=4,
        llm_tool_stream_retry_base_delay_sec=2.0,
        llm_tool_stream_retry_max_delay_sec=60.0,
    )
    await agent.run("hi")

    assert llm.calls == 3
    # attempt 0 -> 2*2**0=2 ; attempt 1 -> 2*2**1=4
    assert delays[:2] == [2.0, 4.0]


# ---------------------------------------------------------------------------
# key_pool._is_retryable_5xx_error / _retry_after_seconds_full
# ---------------------------------------------------------------------------


def test_pool_is_retryable_5xx_classification() -> None:
    assert _is_retryable_5xx_error(_status_error(524))
    assert _is_retryable_5xx_error(_status_error(500))
    assert _is_retryable_5xx_error(
        openai.APIConnectionError(request=httpx.Request("POST", "https://x/v1"))
    )
    # 4xx and rate limits are NOT retryable-5xx (handled by their own branch).
    assert not _is_retryable_5xx_error(_status_error(429))
    assert not _is_retryable_5xx_error(_status_error(400))
    assert not _is_retryable_5xx_error(ValueError("nope"))


def test_pool_retry_after_full_prefers_body_then_response() -> None:
    exc_body = _status_error(524, body={"retry_after": 120})
    assert _retry_after_seconds_full(exc_body, COOLDOWN_SERVER_ERROR) == 120.0

    resp = _http_response(503, {"retry-after": 45})
    exc_resp = openai.InternalServerError("boom", response=resp, body=None)
    assert _retry_after_seconds_full(exc_resp, COOLDOWN_SERVER_ERROR) == 45.0

    # No hint anywhere -> the provided default.
    assert _retry_after_seconds_full(_status_error(503), COOLDOWN_SERVER_ERROR) == (
        COOLDOWN_SERVER_ERROR
    )


# ---------------------------------------------------------------------------
# key_pool: all-keys 5xx wait-and-retry
# ---------------------------------------------------------------------------


class _CountingLLM:
    """Fails with a retryable 5xx for the first ``fail_calls`` invocations."""

    def __init__(self, name: str, *, fail_calls: int, retry_after: float | None = None) -> None:
        self.name = name
        self.base_url = "https://llm.example/v1"
        self.api_key = name
        self.model = "dummy"
        self.fail_calls = fail_calls
        self.retry_after = retry_after
        self.calls = 0
        self._last_call_input_tokens = 1
        self._last_call_output_tokens = 1
        self._last_call_input_cached_tokens = 0
        self._last_call_ttft = 0.1
        self._last_call_tpot = 1.0
        self._last_finish_reason = "stop"

    async def ask(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_calls:
            body = {"retry_after": self.retry_after} if self.retry_after is not None else None
            raise _status_error(524, body=body)
        return self.name


@pytest.mark.asyncio
async def test_pool_waits_and_retries_when_all_keys_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every key 524s on the first sweep; pool waits the cooldown then recovers."""
    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.0)
    slept: list[float] = []

    async def _fast_sleep(delay: float) -> None:
        slept.append(float(delay))

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    # Keep the per-call deadline generous but the cooldown tiny so the test is fast.
    monkeypatch.setenv("SCIENCEFLOW_POOL_5XX_RETRY_BUDGET_SEC", "30")
    monkeypatch.setenv("SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC", "30")

    a = _CountingLLM("a", fail_calls=1, retry_after=0.01)
    b = _CountingLLM("b", fail_calls=1, retry_after=0.01)
    pool = PooledLLM(
        [a, b],
        routing_mode="sticky_failover",
        sticky_primary_index=0,
        server_error_cooldown_sec=0.01,
    )

    result = await pool.ask([])

    assert result in {"a", "b"}
    # Both keys were tried, both cooled down, then a re-sweep succeeded.
    assert a.calls >= 1 and b.calls >= 1
    assert any(d > 0 for d in slept), "pool should have slept while waiting out the 5xx cooldown"


@pytest.mark.asyncio
async def test_pool_raises_when_5xx_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanent 5xx with a zero budget surfaces the error instead of hanging."""
    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.0)

    async def _fast_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    # A tiny budget (< the 0.05s slip floor) lets the first sweep run but makes
    # the all-keys wait branch unable to slip, so the real 5xx surfaces.
    monkeypatch.setenv("SCIENCEFLOW_POOL_5XX_RETRY_BUDGET_SEC", "0.03")
    monkeypatch.setenv("SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC", "0.03")

    a = _CountingLLM("a", fail_calls=99, retry_after=5.0)
    b = _CountingLLM("b", fail_calls=99, retry_after=5.0)
    pool = PooledLLM(
        [a, b],
        routing_mode="sticky_failover",
        sticky_primary_index=0,
        server_error_cooldown_sec=5.0,
    )

    with pytest.raises(openai.APIStatusError):
        await pool.ask([])


@pytest.mark.asyncio
async def test_pool_5xx_skips_soft_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retryable 5xx must not trigger the same-key soft retry (no double sleep)."""
    soft_retry_calls = {"n": 0}
    original_sleep = asyncio.sleep

    async def _counting_sleep(delay: float) -> None:
        soft_retry_calls["n"] += 1
        # Do not actually wait.
        await original_sleep(0)

    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.5)
    monkeypatch.setattr(asyncio, "sleep", _counting_sleep)
    # Tiny budget so the first sweep runs but the all-keys wait cannot slip.
    monkeypatch.setenv("SCIENCEFLOW_POOL_5XX_RETRY_BUDGET_SEC", "0.03")
    monkeypatch.setenv("SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC", "0.03")

    # Single key, always 524 -> raises after one sweep (no soft retry, no wait).
    a = _CountingLLM("a", fail_calls=99, retry_after=5.0)
    pool = PooledLLM([a], server_error_cooldown_sec=5.0)

    with pytest.raises(openai.APIStatusError):
        await pool.ask([])

    # The 5xx path skips the soft retry, so the 0.5s soft-retry sleep never runs.
    # (a.calls stays 1: one attempt, no same-key soft retry.)
    assert a.calls == 1


# ---------------------------------------------------------------------------
# solver._worker_error_kind reclassification
# ---------------------------------------------------------------------------


def test_worker_error_kind_classifies_5xx_as_llm_api_error() -> None:
    # The run3 524: InternalServerError text contains "timeout occurred"; it must
    # be llm_api_error, NOT worker_timeout.
    assert (
        LnrSolver._worker_error_kind(
            "InternalServerError: Error code: 524 - a timeout occurred"
        )
        == "llm_api_error"
    )
    assert (
        LnrSolver._worker_error_kind("Error code: 503 - Service Unavailable")
        == "llm_api_error"
    )
    assert LnrSolver._worker_error_kind("APIStatusError: 502 bad gateway") == "llm_api_error"


def test_worker_error_kind_preserves_existing_classes() -> None:
    # A genuine worker timeout (no 5xx marker) is still worker_timeout.
    assert LnrSolver._worker_error_kind("worker timeout after 43500s") == "worker_timeout"
    assert LnrSolver._worker_error_kind("ReadError: stream disconnected") == (
        "llm_transport_error"
    )
    assert LnrSolver._worker_error_kind("Error code: 402 - insufficient balance") == (
        "llm_quota_error"
    )
    assert LnrSolver._worker_error_kind("") == ""
