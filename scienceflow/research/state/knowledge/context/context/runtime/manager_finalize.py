"""MemoryContextManager responsibility: finalize."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
)
from scienceflow.research.state.knowledge.context.context.runtime.finalize_runtime import (
    _build_compaction_prompt,
    _compaction_history,
    _install_compaction_summary,
    _pinned_leading_messages,
)


def inject_file_content(self, path: str, *, reason: str = "") -> str:
    resolved = self._guard.resolve(path)
    if not resolved.is_file():
        raise OSError(f"not a file: {path}")
    text = resolved.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header = f"[auto-read: {reason}]\n" if reason else "[auto-read]\n"
    if len(lines) > 300:
        preview = (
            "\n".join(lines[:150])
            + f"\n... ({len(lines)} lines total) ...\n"
            + "\n".join(lines[-50:])
        )
    else:
        preview = text
    return header + preview


async def compact(
    self,
    *,
    mid_run: bool = False,
    completion: Callable[[str, str], Awaitable[str]],
) -> str:
    """Compact chat while preserving the user pin and protected EDA prefix."""
    messages = _all_messages(self._memory)
    if not messages:
        return "Nothing to compact."
    pinned = _pinned_leading_messages(self, messages)
    history, protected, protected_chars = _compaction_history(self, messages, pinned)
    prompt, system_prompt = _build_compaction_prompt(
        history,
        protected=protected,
        protected_chars=protected_chars,
        mid_run=mid_run,
    )
    summary = await completion(prompt, system_prompt)
    if not (summary or "").strip():
        return "Compact failed: empty summary."
    _install_compaction_summary(
        self,
        pinned=pinned,
        protected=protected,
        summary=summary,
    )
    return f"Compacted {len(messages)} messages into summary ({len(summary)} chars)."
