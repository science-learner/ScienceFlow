# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Rebuildable agent and stage-memory projections."""

from scienceflow.research.state.knowledge.memory.core.policy import (
    LnrContextMemoryPolicyResult,
    ensure_lnr_context_memory_budget,
)
from scienceflow.research.state.knowledge.memory.core.service import MemoryService
from scienceflow.research.state.knowledge.memory.core.compactor import MemoryCompactor
from scienceflow.research.state.knowledge.memory.core.contracts import (
    MemoryCompactionRequest,
    MemoryCompactionResult,
    ProtectedContextRequest,
    ProtectedContextResult,
)
from scienceflow.research.state.knowledge.memory.context.protected_context import ProtectedContextAdapter
from scienceflow.research.state.knowledge.memory.records.stage_memory import StageMemoryCard, StageMemoryView

__all__ = [
    "LnrContextMemoryPolicyResult",
    "MemoryCompactionRequest",
    "MemoryCompactionResult",
    "MemoryCompactor",
    "ProtectedContextAdapter",
    "ProtectedContextRequest",
    "ProtectedContextResult",
    "StageMemoryCard",
    "StageMemoryView",
    "MemoryService",
    "ensure_lnr_context_memory_budget",
]
