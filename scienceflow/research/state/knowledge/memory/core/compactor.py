# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Replayable compactor for InquiryCraft-compatible memory record files."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from scienceflow.research.state.knowledge.memory.records.agent_records import (
    select_prefix_safe_agent_memory_records,
    write_agent_memory_record_files,
)
from scienceflow.research.state.knowledge.memory.core.contracts import (
    MemoryCompactionRequest,
    MemoryCompactionResult,
)


class MemoryCompactor:
    """Own record selection and replacement, but not prompt construction."""

    def compact(self, request: MemoryCompactionRequest) -> MemoryCompactionResult:
        source = Path(request.source_dir)
        strength = "strict_context_limit" if request.strict_context_limit else "normal"
        if not source.is_dir():
            return MemoryCompactionResult(
                applied=False,
                reason="missing_source_memory",
                compact_strength=strength,
                metadata=request.metadata,
            )
        max_messages = max(1, int(request.max_messages or 1))
        records = select_prefix_safe_agent_memory_records(
            source,
            max_messages=min(max_messages, 8) if request.strict_context_limit else max_messages,
        )
        if request.strict_context_limit:
            records = self.minimal_task_prefix(records)
        if not records:
            return MemoryCompactionResult(
                applied=False,
                reason="empty_source_memory",
                compact_strength=strength,
                metadata=request.metadata,
            )
        source_count = len(records)
        projected = [*records, self.user_record(request.prompt)]
        destination = Path(request.destination_dir)
        try:
            if destination.exists():
                shutil.rmtree(destination)
            write_agent_memory_record_files(
                destination,
                projected,
                write_long_term=True,
            )
        except OSError as exc:
            return MemoryCompactionResult(
                applied=False,
                reason=f"write_failed:{type(exc).__name__}",
                source_record_count=source_count,
                compact_strength=strength,
                metadata=request.metadata,
            )
        return MemoryCompactionResult(
            applied=True,
            reason="compacted",
            record_count=len(projected),
            source_record_count=source_count,
            projection_hash=self.projection_hash(projected),
            compact_strength=strength,
            metadata=request.metadata,
        )

    @staticmethod
    def user_record(content: str) -> dict[str, Any]:
        return {
            "uuid": str(uuid.uuid4()),
            "message": {
                "__class__": "Message",
                "role": "user",
                "content": str(content or ""),
            },
            "role": "user",
            "extra_info": {},
            "timestamp": time.time(),
            "agent_id": "",
        }

    @classmethod
    def minimal_task_prefix(
        cls, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        prefix: list[dict[str, Any]] = []
        for record in records:
            if cls.record_role(record) in {"system", "user"}:
                prefix.append(record)
                continue
            break
        if prefix:
            return prefix
        for record in records:
            if cls.record_role(record) in {"system", "user"}:
                return [record]
        return []

    @staticmethod
    def record_role(record: dict[str, Any]) -> str:
        message = record.get("message")
        if isinstance(message, dict):
            return str(message.get("role") or record.get("role") or "").strip()
        return str(record.get("role") or "").strip()

    @staticmethod
    def projection_hash(records: list[dict[str, Any]]) -> str:
        semantic: list[dict[str, Any]] = []
        for record in records:
            message = record.get("message")
            semantic.append(
                {
                    "role": MemoryCompactor.record_role(record),
                    "message": message if isinstance(message, dict) else {},
                    "extra_info": record.get("extra_info") or {},
                }
            )
        payload = json.dumps(semantic, ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["MemoryCompactor"]
