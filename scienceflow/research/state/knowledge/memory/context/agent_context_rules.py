"""Pure helpers shared by the run-loop responsibility modules."""

from __future__ import annotations

from typing import Any

from inquirycraft.memory import Message


_LNR_FIRST_TASK_PROMPT_PREFIXES = (
    "You are solving one ML task in a continuous REPL workspace.",
    "You are solving one optimization task in a continuous REPL workspace.",
)


def _is_lnr_first_task_prompt_text(text: str | None) -> bool:
    clean = str(text or "").lstrip()
    return any(clean.startswith(prefix) for prefix in _LNR_FIRST_TASK_PROMPT_PREFIXES)


def _message_text(message: Message) -> str:
    return (
        message.content
        if isinstance(message.content, str)
        else str(message.content or "")
    )


def _clip_stage_commit_context_text(text: str, *, max_chars: int) -> str:
    clean = str(text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    marker = "\n...[stage context clipped]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return clean[:head].rstrip() + marker + clean[-tail:].lstrip()


def _is_stage_commit_memory_text(text: str) -> bool:
    clean = str(text or "")
    return (
        "[stage append-only write]" in clean
        or "[LNR_STAGE_COMMIT_REQUEST]" in clean
        or ("STAGE_COMMIT_BEGIN" in clean and "STAGE_COMMIT_END" in clean)
        or (
            "```json" in clean.lower()
            and '"stage_id"' in clean
            and '"metric_validity"' in clean
            and '"files"' in clean
        )
    )


def _memory_has_lnr_first_task_prompt(memory: Any) -> bool:
    try:
        records = memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return False
    for record in records or []:
        try:
            message = record.memory_record.message
        except Exception:
            continue
        if getattr(message, "role", None) == "user" and _is_lnr_first_task_prompt_text(
            _message_text(message)
        ):
            return True
    return False


def _should_append_run_request(memory: Any, request: str | None) -> bool:
    if not request:
        return False
    if _is_lnr_first_task_prompt_text(request) and _memory_has_lnr_first_task_prompt(
        memory
    ):
        return False
    return True


__all__ = (
    "_clip_stage_commit_context_text",
    "_is_lnr_first_task_prompt_text",
    "_is_stage_commit_memory_text",
    "_message_text",
    "_should_append_run_request",
)
