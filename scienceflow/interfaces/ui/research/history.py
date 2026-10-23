"""Research display folding and a session-local record of lifecycle events."""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_TASK_EVENT = re.compile(r"^Task (\d+) · ")
_RUN_ADDRESS = re.compile(r" · run \S+$")


def _visible_activity(events: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Show the latest state of each task while retaining the full event file."""
    latest: dict[int, tuple[int, int, str]] = {}
    standalone: list[tuple[int, int, str]] = []
    preparing: int | None = None
    for order, (position, summary) in enumerate(events):
        match = _TASK_EVENT.match(summary)
        if match:
            number = int(match.group(1))
            if summary.endswith("preparation started"):
                preparing = number
            elif preparing == number and " · run " in summary:
                preparing = None
            latest[number] = (position, order, _RUN_ADDRESS.sub("", summary))
            continue
        if preparing is not None and summary in {
            "Long research · preparation cancelled",
            "Long research · preflight failed",
        }:
            latest[preparing] = (
                position,
                order,
                f"Task {preparing} · {summary.partition(' · ')[2]}",
            )
            if summary.endswith("cancelled"):
                preparing = None
            continue
        standalone.append((position, order, summary))
    visible = sorted((*standalone, *latest.values()), key=lambda event: (event[0], event[1]))
    return [(position, summary) for position, _, summary in visible]


class ResearchHistory:
    def __init__(self) -> None:
        self.section = None
        self._restored: dict[int, list[str]] = defaultdict(list)

    @staticmethod
    def _activity_path(host) -> Path | None:
        options = getattr(getattr(host, "runtime", None), "options", None)
        session_log = getattr(options, "session_log", None)
        return (
            Path(session_log).with_name("scienceflow_activity.jsonl")
            if session_log
            else None
        )

    async def load(self, host) -> None:
        """Load only concise SF events; the chat conversation remains untouched."""
        path = self._activity_path(host)
        if path is None or not path.is_file():
            return

        def read_events():
            with path.open(encoding="utf-8") as stream:
                return deque(stream, maxlen=200)

        try:
            lines = await asyncio.to_thread(read_events)
            events = []
            for line in lines:
                try:
                    event = json.loads(line)
                    position = int(event["message_index"])
                    summary = str(event["summary"])
                except (ValueError, TypeError, KeyError):
                    continue
                if position >= 0 and summary:
                    events.append((position, summary))
            for position, summary in _visible_activity(events):
                self._restored[position].append(summary)
        except OSError:
            logger.warning("Could not read ScienceFlow session activity", exc_info=True)

    def history_before(self, message_index: int) -> tuple[str, ...]:
        return tuple(self._restored.pop(message_index, ()))

    async def show_legacy_history(self, host) -> None:
        """Older InquiryCraft hosts lack ordered replay; show a concise fallback."""
        events = [summary for rows in self._restored.values() for summary in rows]
        if events:
            await host.notice(
                "Previous ScienceFlow activity:\n" + "\n".join(events[-8:])
            )
        self._restored.clear()

    async def record(self, host, summary: str) -> None:
        path = self._activity_path(host)
        if path is None:
            return
        conversation = getattr(getattr(host, "runtime", None), "conversation", None)
        position = len(getattr(conversation, "messages", ()))
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "message_index": position,
            "summary": " ".join(summary.split())[:240],
        }

        def append_event():
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = (
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

        try:
            await asyncio.to_thread(append_event)
        except OSError:
            logger.warning("Could not save ScienceFlow session activity", exc_info=True)

    async def begin(self, host) -> None:
        self.section = None
        begin = getattr(host, "begin_transcript_section", None)
        fold = getattr(host, "fold_transcript_section", None)
        if callable(begin) and callable(fold):
            try:
                self.section = await begin()
            except (RuntimeError, ValueError):
                logger.warning(
                    "Could not mark research configuration display", exc_info=True
                )

    async def fold(self, host, draft, label: str) -> bool:
        code, feedback = ",".join(draft.code_models), ",".join(draft.feedback_models)
        models = code if code == feedback else f"Code {code} · Feedback {feedback}"
        summary = (
            f"{label} · Configuration · {draft.workers} workers · CPU {draft.cpu_list}"
            f" · GPU {draft.gpu_list} · {(draft.wall_clock_sec or 0) / 60:g} min"
            + (f" · {models}" if models else "")
        )
        return await self.fold_summary(host, summary)

    async def fold_summary(self, host, summary: str) -> bool:
        """Close the current preparation section after launch or cancellation."""
        section, self.section = self.section, None
        fold = getattr(host, "fold_transcript_section", None)
        if section is None or not callable(fold):
            return False
        try:
            return bool(await fold(section, summary))
        except (RuntimeError, ValueError):
            # A presentation failure must not turn a successful launch into a
            # failed task or encourage launching the same task again.
            logger.warning(
                "Could not fold research history display", exc_info=True
            )
            return False
