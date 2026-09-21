# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Adapter preserving the established LNR stage-memory rendering algorithm."""

from scienceflow.research.state.knowledge.memory.records.stage_memory import (
    StageMemoryView,
    build_stage_memory_view,
    sync_current_segment,
)

__all__ = ["StageMemoryView", "build_stage_memory_view", "sync_current_segment"]
