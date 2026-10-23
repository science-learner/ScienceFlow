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

"""ScienceFlow compatibility facade for InquiryCraft resume inspection."""

from __future__ import annotations

from typing import Any

from inquirycraft.runtime import (
    PendingToolCall as PendingToolCall,
    ResumeState,
    inspect_resume_messages,
)

ResumeMemoryState = ResumeState


def inspect_resume_memory_messages(messages_in: list[Any]) -> ResumeMemoryState:
    return inspect_resume_messages(messages_in)


def inspect_agent_resume_state(agent: Any) -> ResumeMemoryState:
    try:
        records = agent.memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return ResumeMemoryState(
            action="fresh_start",
            reason="memory_unreadable",
            message_count=0,
            last_role="",
        )
    return inspect_resume_memory_messages(list(records or []))
