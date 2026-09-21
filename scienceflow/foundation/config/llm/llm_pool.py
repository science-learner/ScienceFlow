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

"""ScienceFlow configuration facade for the InquiryCraft LLM pool.

The failover state machine is domain-independent and lives in InquiryCraft. This
module supplies ScienceFlow's logger, environment variable, client-construction hook,
constants, and soft-retry tuning.
"""

from __future__ import annotations

import logging
from typing import Any

from inquirycraft.llm import (
    KeyPool as InquiryCraftKeyPool,
    PooledLLM as InquiryCraftPooledLLM,
    OnlineLLM,
    parse_key_env as parse_key_env,
)


logger = logging.getLogger("scienceflow")

COOLDOWN_RATE_LIMIT = 60.0
COOLDOWN_CONNECTION = 15.0
_DEFAULT_POOL_RL_MAX_WAIT_SEC = 300.0
_SOFT_RETRY_DELAY_SEC = 0.5


class KeyPool(InquiryCraftKeyPool):
    """ScienceFlow logging facade over the InquiryCraft endpoint pool."""

    def __init__(self, endpoints: list[tuple[str, str]]) -> None:
        super().__init__(endpoints, logger=logger)


class PooledLLM(InquiryCraftPooledLLM):
    """ScienceFlow-configured InquiryCraft failover pool."""

    runtime_logger = logger
    rate_limit_max_wait_env = "SCIENCEFLOW_POOL_RL_MAX_WAIT_SEC"
    default_rate_limit_max_wait_sec = _DEFAULT_POOL_RL_MAX_WAIT_SEC

    @classmethod
    def _create_online_llm(cls, **kwargs: Any) -> OnlineLLM:
        # Resolve this module symbol at call time. Existing extensions and tests
        # replace this module's ``OnlineLLM`` during construction.
        return OnlineLLM(**kwargs)

    def _create_key_pool(self, endpoints: list[tuple[str, str]]) -> KeyPool:
        return KeyPool(endpoints)

    def _soft_retry_delay_sec(self) -> float:
        # Keep ScienceFlow's configured soft-retry delay.
        return _SOFT_RETRY_DELAY_SEC


__all__ = ["KeyPool", "PooledLLM", "parse_key_env"]
