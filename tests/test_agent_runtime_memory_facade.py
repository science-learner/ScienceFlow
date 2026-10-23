from __future__ import annotations

import json
from types import SimpleNamespace

from inquirycraft.memory import Function, Message, ToolCall

from scienceflow.research.state.knowledge.memory.records.agent_records import (
    _prune_memory_to_recent_rounds,
    rewrite_workspace_abs_to_relative,
    sanitize_inherited_memory_workspace_paths,
)


class Memory:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = list(messages)
        self.storage = self.messages
        self.chat_history_memory = self

    def retrieve(self, window_size=None):
        _ = window_size
        return [
            SimpleNamespace(memory_record=SimpleNamespace(message=message))
            for message in self.messages
        ]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)


def test_scienceflow_workspace_path_facade_preserves_exact_text_contract() -> None:
    workspace = "/tmp/runtime/workspace"
    assert rewrite_workspace_abs_to_relative(
        f"read {workspace}/solution.py then cd {workspace}",
        workspace,
    ) == "read ./solution.py then cd ."


def test_scienceflow_inherited_memory_policy_rewrites_content_and_tool_arguments() -> None:
    stale = "/tmp/run/wsp/0123456789abcdef0123456789abcdef/solution.py"
    call = ToolCall(
        id="call-1",
        function=Function(
            name="read",
            arguments=json.dumps({"path": stale}),
        ),
    )
    memory = Memory(
        [
            Message.user_message(f"inspect {stale}"),
            Message(role="assistant", content="", tool_calls=[call]),
        ]
    )

    sanitize_inherited_memory_workspace_paths(memory)

    assert memory.messages[0].content == "inspect ./solution.py"
    assert json.loads(memory.messages[1].tool_calls[0].function.arguments) == {
        "path": "./solution.py"
    }


def test_scienceflow_recent_round_facade_keeps_historical_prefix() -> None:
    memory = Memory(
        [
            Message.system_message("system"),
            Message.user_message("task"),
            Message.assistant_message("old"),
            Message.tool_message("old result", "bash", "old"),
            Message.assistant_message("new"),
            Message.tool_message("new result", "bash", "new"),
        ]
    )

    _prune_memory_to_recent_rounds(memory, 1)

    assert [(message.role, message.content) for message in memory.messages] == [
        ("system", "system"),
        ("user", "task"),
        ("assistant", "new"),
        ("tool", "new result"),
    ]
