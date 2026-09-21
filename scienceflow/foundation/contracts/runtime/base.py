# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Shared version metadata for cross-module ScienceFlow contracts."""

from __future__ import annotations

from typing import ClassVar


class VersionedContract:
    """Marker base for contracts that cross independently evolving modules.

    The version is intentionally a class variable so adding contract metadata
    does not change legacy dataclass constructors or serialized payloads.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    @classmethod
    def contract_schema_version(cls) -> str:
        return cls.SCHEMA_VERSION
