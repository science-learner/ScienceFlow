"""Stateless chat-compaction helpers for ``MemoryContextManager``."""

from __future__ import annotations

from typing import Any

from inquirycraft.memory import Message

from scienceflow.research.state.knowledge.context.context.results.base import COMPACTED_CONVERSATION_SUMMARY_MARKER
from scienceflow.research.state.knowledge.context.context.compression.clone_compression import _clear_memory_and_long_term_log
from scienceflow.research.state.knowledge.context.context.compression.compaction import (
    MSG0_BODY_TRUNCATED_PLACEHOLDER,
    _format_message_excerpt_for_compact,
    _split_msg0_head_body,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _message_text_for_chars,
    _message_with_text,
)

def _pinned_leading_messages(manager: Any, messages: list[Message]) -> list[Message]:
    pinned: list[Message] = []
    for index, message in enumerate(messages):
        if getattr(message, "role", None) != "user":
            break
        text = _message_text_for_chars(message)
        if index == 0 and manager._msg0_compress_body:
            head, body = _split_msg0_head_body(text)
            if body:
                pinned.append(
                    _message_with_text(
                        message,
                        head.rstrip("\n")
                        + "\n\n## Task description\n"
                        + MSG0_BODY_TRUNCATED_PLACEHOLDER,
                    )
                )
                continue
        pinned.append(message)
    return pinned


def _append_workspace_ground_truth(manager: Any, history_text: str) -> str:
    solution_path = manager._workspace / "solution.py"
    if solution_path.is_file():
        try:
            solution = solution_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            solution = ""
        if solution.strip():
            if len(solution) > 6000:
                solution = solution[:6000] + "\n... [truncated]"
            history_text += (
                "\n\n[CURRENT solution.py — MUST be reflected in the summary]\n"
                "```python\n" + solution + "\n```"
            )
    result_path = manager._workspace / "result.md"
    if result_path.is_file():
        try:
            result_text = result_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            result_text = ""
        if result_text.strip():
            history_text += "\n\n[CURRENT result.md]\n" + result_text[:3000]
    return history_text


def _compaction_history(
    manager: Any,
    messages: list[Message],
    pinned: list[Message],
) -> tuple[str, list[Message], int]:
    protected, source_messages, protected_chars = manager._protected_raw_prefix_parts(
        messages, len(pinned)
    )
    source = [*pinned, *source_messages] if protected else messages
    history = "\n".join(
        f"[{message.role}]: {_format_message_excerpt_for_compact(message)}"
        for message in source
    )
    return _append_workspace_ground_truth(manager, history), protected, protected_chars


def _compaction_pin_guard(protected: list[Message], protected_chars: int) -> str:
    protected_guard = ""
    if protected:
        protected_guard = (
            f"- {len(protected)} early EDA/tool message(s) are also re-attached verbatim "
            f"after compact ({protected_chars} chars); do NOT repeat or paraphrase them.\n"
        )
    return (
        "USER PIN (preserved verbatim by the system — do NOT repeat or paraphrase here):\n"
        "- The first user message's head (everything before the line `## Task description`) "
        "is re-attached verbatim after this summary; you do NOT need to include it.\n"
        + protected_guard
        + "- Treat prohibitions in that pin as **phase-scoped** unless the recent chat "
        "explicitly says otherwise: e.g. \"do **NOT** ensemble across models\" applies to "
        "**single-model training** (explore / exploit_explore), not after a later message "
        "switches the workflow to **ensemble** peer fusion.\n"
        "- Do NOT drop, soften, or rewrite explicit user constraints / prohibitions written "
        "there when they still apply to the **current** phase. If the recent chat shows the "
        "agent violating an applicable constraint, call it out in a section titled "
        "`User constraints (verbatim)` and quote each violated line word-for-word.\n\n"
    )


def _build_compaction_prompt(
    history: str,
    *,
    protected: list[Message],
    protected_chars: int,
    mid_run: bool,
) -> tuple[str, str]:
    guard = _compaction_pin_guard(protected, protected_chars)
    system_suffix = (
        "The user pin (msg[0] head) is re-attached verbatim by the system, so do not repeat it; "
        "do not soften or drop explicit user prohibitions."
    )
    if mid_run:
        prompt = (
            guard
            + "The chat history below no longer fits in the model context window. Summarize it so the "
            "**same** agent can continue debugging and improving in this session.\n\n"
            "IMPORTANT: Reflect the **current** `solution.py` (pipeline, model, key hyperparameters) "
            "and any metrics in ``[CURRENT result.md]``. Preserve recent **errors, tracebacks, and "
            "failed bash runs** verbatim enough to fix them.\n\n"
            "Use markdown headings:\n"
            "1. **Current approach** — data loading, features, model, validation.\n"
            "2. **Recent failures** — last errors, exit codes, stderr highlights.\n"
            "3. **What to do next** — concrete fixes or checks (files, commands).\n"
            "4. **User constraints (verbatim)** — quote only prohibitions from the user pin that "
            "**still apply** to the current phase (see phase-scoping note above); omit this "
            "section if none apply.\n\n"
            "Max ~600 words. Be specific (paths, function names, numbers).\n\n"
            + history
        )
        return prompt, "You compress prior chat into a short continuation summary for the same debugging session. " + system_suffix
    prompt = (
        guard
        + "Summarize the following conversation into a structured handoff document for the NEXT agent "
        "that will continue this workspace. Include these sections (use markdown headings):\n\n"
        "IMPORTANT: Your summary MUST accurately reflect the **current** `solution.py` (model family, "
        "key hyperparameters, main training logic) and any metric values in ``[CURRENT result.md]`` "
        "or the chat history. Do not claim \"no modeling\" or \"not yet implemented\" if the files "
        "below show otherwise.\n\n"
        "1. **Data insights** — column types, missing patterns, key correlations, train/val split notes.\n"
        "2. **Feature engineering pipeline** — encoding, imputation, scaling, derived features; name key functions.\n"
        "3. **Model architecture** — algorithms, main hyperparameters, training setup.\n"
        "4. **What was tried and results** — approaches attempted, metric values (quick vs full if known), what failed.\n"
        "5. **Pitfalls** — bugs, leakage risks, NaNs, timeouts.\n"
        "6. **Recommended next steps** — concrete improvements.\n"
        "7. **User constraints (verbatim)** — quote only prohibitions from the user pin that "
        "**still apply** to the current phase (see phase-scoping note above); omit this section if none apply.\n\n"
        "Max ~1000 words. Be concise but specific: prefer function names, column lists, and numbers over vague prose.\n\n"
        + history
    )
    return prompt, "You compress prior chat into a durable summary. " + system_suffix


def _install_compaction_summary(
    manager: Any,
    *,
    pinned: list[Message],
    protected: list[Message],
    summary: str,
) -> None:
    _clear_memory_and_long_term_log(manager._memory)
    for message in [*pinned, *protected]:
        manager._memory.add_message(message)
    manager._memory.add_message(
        Message.user_message(f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\n{summary.strip()}")
    )
    if protected:
        manager._protected_raw_prefix_end_index = len(pinned) + len(protected)
