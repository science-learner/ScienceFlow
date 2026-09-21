# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned contracts for durable agent-memory compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MemoryCompactionRequest:
    source_dir: str
    destination_dir: str
    prompt: str
    max_messages: int = 100
    strict_context_limit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class MemoryCompactionResult:
    applied: bool
    reason: str
    record_count: int = 0
    source_record_count: int = 0
    projection_hash: str = ""
    compact_strength: str = "normal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class ProtectedContextRequest:
    end_index: int
    mode: str = "facts"
    warn_chars: int = 50_000
    summary: str = ""
    captured_before_commit: bool = False

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class ProtectedContextResult:
    applied: bool
    reason: str
    protected_end_index: int = 0
    info: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


__all__ = [
    "MemoryCompactionRequest",
    "MemoryCompactionResult",
    "ProtectedContextRequest",
    "ProtectedContextResult",
]
