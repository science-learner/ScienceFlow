# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned contracts emitted by the read-only observation plane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    schema_version: str
    sequence: int
    kind: str
    source: str
    timestamp: float
    run_id: str = ""
    worker_id: str = ""
    correlation_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["ObservationEnvelope"]
