from __future__ import annotations

from inquirycraft.llm import OnlineLLM
from inquirycraft.memory import Message
from scienceflow.foundation.config.schema.settings import Config, _apply_env


def _llm(*, coalesce: bool) -> OnlineLLM:
    return OnlineLLM(
        model="test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        coalesce_system_messages=coalesce,
    )


def test_system_messages_remain_layered_by_default() -> None:
    llm = _llm(coalesce=False)
    result = llm._format_request_messages(
        [Message.user_message("task")],
        [Message.system_message("stable"), Message.system_message("runtime")],
    )

    assert [row["role"] for row in result] == ["system", "system", "user"]
    assert [row["content"] for row in result] == ["stable", "runtime", "task"]


def test_vllm_compat_coalesces_only_provider_payload() -> None:
    system_msgs = [Message.system_message("stable"), Message.system_message("runtime")]
    llm = _llm(coalesce=True)

    result = llm._format_request_messages([Message.user_message("task")], system_msgs)

    assert [row["role"] for row in result] == ["system", "user"]
    assert result[0]["content"] == "stable\n\nruntime"
    assert [msg.content for msg in system_msgs] == ["stable", "runtime"]


def test_coalesce_system_messages_env_is_stage_specific(monkeypatch) -> None:
    monkeypatch.setenv("CODE_COALESCE_SYSTEM_MESSAGES", "true")
    monkeypatch.setenv("FEEDBACK_COALESCE_SYSTEM_MESSAGES", "false")
    cfg = Config()

    _apply_env(cfg)

    assert cfg.agent.code.coalesce_system_messages is True
    assert cfg.agent.feedback.coalesce_system_messages is False
