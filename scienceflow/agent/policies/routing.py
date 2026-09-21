"""Run-loop responsibility: routing."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from inquirycraft.memory import Message
from inquirycraft.tools import tool_call_arguments_dict, tool_call_function_name

from scienceflow.agent.policies.repl import (
    looks_like_repl_small_talk,
)
from scienceflow.agent.policies.repl import (
    tool_choice_for_main_loop as _tool_choice_for_main_loop_fn,
)
from scienceflow.research.state.knowledge.memory.agent.memory_utils import (
    _tool_call_names_from_list,
)
from scienceflow.research.state.knowledge.memory.agent.resource_feedback_memory import (
    RESOURCE_STATE_SUMMARY_MARKER,
)
from scienceflow.research.state.knowledge.memory.context.agent_context_rules import (
    _clip_stage_commit_context_text,
    _is_lnr_first_task_prompt_text,
    _is_stage_commit_memory_text,
    _message_text,
)

logger = logging.getLogger("scienceflow")


def _agentic_route_log_dir(self) -> Path:
    override = getattr(self, "_agentic_route_log_dir_override", None)
    if override:
        return Path(override)
    return self._workspace_dir / ".logs"


def _write_agentic_route_response(self, text: str) -> None:
    body = str(text or "").strip()
    if not body:
        return
    try:
        path = self._agentic_route_log_dir() / "agentic_route_response.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.rstrip() + "\n", encoding="utf-8")
    except OSError:
        logger.debug("[agentic-route] route response capture skipped", exc_info=True)


def _build_lnr_stage_commit_compact_messages(
    self,
    *,
    base_messages: list[Message],
    transient_user_prompt: str,
) -> list[Message]:
    task_context = ""
    recent_parts: list[str] = []
    recent_budget = int(
        getattr(self, "_lnr_stage_commit_recent_context_chars", 7000) or 7000
    )
    per_message_budget = int(
        getattr(self, "_lnr_stage_commit_message_chars", 1200) or 1200
    )

    for msg in base_messages:
        text = _message_text(msg).strip()
        if not text:
            continue
        if getattr(msg, "role", None) == "user" and _is_lnr_first_task_prompt_text(
            text
        ):
            task_context = _clip_stage_commit_context_text(text, max_chars=3000)
            break

    used = 0
    for msg in reversed(base_messages):
        role = str(getattr(msg, "role", "") or "")
        if role == "system":
            continue
        text = _message_text(msg).strip()
        if (
            not text
            or _is_lnr_first_task_prompt_text(text)
            or _is_stage_commit_memory_text(text)
        ):
            continue
        clipped = _clip_stage_commit_context_text(text, max_chars=per_message_budget)
        if not clipped:
            continue
        part = f"{role or 'message'}:\n{clipped}"
        if used + len(part) > recent_budget and recent_parts:
            break
        recent_parts.append(part)
        used += len(part)

    messages: list[Message] = []
    if task_context:
        messages.append(
            Message.user_message("[LNR_STAGE_COMMIT_TASK_CONTEXT]\n" + task_context)
        )
    if recent_parts:
        recent_parts.reverse()
        messages.append(
            Message.user_message(
                "[LNR_STAGE_COMMIT_RECENT_CONTEXT]\n"
                "This is a clipped recent context window for stage bookkeeping only.\n\n"
                + "\n\n".join(recent_parts),
            ),
        )
    messages.append(Message.user_message(transient_user_prompt))
    return messages


def _sync_resource_state_summary_slot(self) -> None:
    """Remove legacy resource summary slots from main-agent chat memory."""
    try:
        records = self.memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return
    kept: list[Message] = []
    removed = 0
    for record in records or []:
        try:
            message = record.memory_record.message
        except Exception:
            continue
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content or "")
        )
        if content.startswith(RESOURCE_STATE_SUMMARY_MARKER):
            removed += 1
            continue
        kept.append(message)
    self._resource_state_summary_last_text = ""
    if removed <= 0:
        return
    try:
        rewrite = getattr(getattr(self, "_memory_ctx", None), "rewrite_messages", None)
        if callable(rewrite):
            rewrite(kept)
        else:
            self.memory.chat_history_memory.storage.clear()
            for message in kept:
                self.memory.add_message(message)
        logger.info(
            "[resource-state] removed %d legacy memory summary slot(s)", removed
        )
    except Exception:
        logger.debug(
            "[resource-state] legacy summary slot cleanup skipped", exc_info=True
        )


def _append_agentic_route_decision_log(
    self,
    *,
    prompt: str,
    response: str,
    trigger: str,
    base_messages_count: int,
    correlation: dict[str, Any] | None = None,
) -> None:
    try:
        path = self._agentic_route_log_dir() / "agentic_route_decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = str(prompt or "")
        response_text = str(response or "")
        record = {
            "event": "agentic_route_decision",
            "timestamp": time.time(),
            "trigger": str(trigger or ""),
            "workspace_node": self._workspace_dir.name,
            "input_compact_sha": hashlib.sha256(
                prompt_text.encode("utf-8", errors="ignore"),
            ).hexdigest(),
            "input_compact_chars": len(prompt_text),
            "input_compact": prompt_text,
            "raw_output": response_text,
            "raw_output_chars": len(response_text),
            "base_messages_count": int(base_messages_count),
            "llm_correlation": dict(correlation or {}),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        logger.debug("[agentic-route] route decision audit log skipped", exc_info=True)


def _accumulate_last_llm_call_tokens_into_run(self) -> None:
    try:
        ti = getattr(self.llm, "_last_call_input_tokens", None)
        to = getattr(self.llm, "_last_call_output_tokens", None)
        tc = getattr(self.llm, "_last_call_input_cached_tokens", None)
        if ti is not None:
            self._run_route_tokens_in += int(ti or 0)
        if to is not None:
            self._run_route_tokens_out += int(to or 0)
        if tc is not None:
            self._run_route_tokens_cached += int(tc or 0)
        self._run_route_llm_calls += 1
    except (TypeError, ValueError):
        pass


async def run_ephemeral_agentic_route_prompt(
    self,
    prompt: str,
    *,
    workspace: Any | None = None,
    trigger: str = "route",
    base_messages: list[Any] | None = None,
    system_messages: list[Any] | None = None,
    timeout: float | None = None,
    llm_override: Any | None = None,
    llm_role: str | None = None,
    stage_id: str | None = None,
    lineage_id: str | None = None,
    node_uid: str | None = None,
) -> str:
    """Ask for a route decision in an isolated, fully audited runtime session."""
    _ = workspace
    route_prompt = str(prompt or "").strip()
    if not route_prompt:
        return ""

    messages = (
        list(base_messages)
        if base_messages is not None
        else self._memory_ctx.build_messages_for_llm()
    )
    from scienceflow.research.control.ephemeral_agent_session import (
        CorrelatedText,
        run_ephemeral_science_agent,
    )

    assistant_msg = await run_ephemeral_science_agent(
        self,
        route_prompt,
        trigger=trigger,
        base_messages=messages,
        system_messages=system_messages,
        timeout=timeout,
        llm_override=llm_override,
        llm_role=llm_role,
        stage_id=stage_id,
        lineage_id=lineage_id,
        node_uid=node_uid,
    )
    text = (getattr(assistant_msg, "content", None) or "").strip()
    if not text:
        text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
    self._write_agentic_route_response(text)
    self._append_agentic_route_decision_log(
        prompt=route_prompt,
        response=text,
        trigger=trigger,
        base_messages_count=len(messages) - 1,
        correlation=assistant_msg.correlation,
    )
    return CorrelatedText(text, assistant_msg.correlation)


def _looks_like_repl_small_talk(self, text: str) -> bool:
    return looks_like_repl_small_talk(text)


def _tool_choice_for_main_loop(self, round_idx: int, run_request: str | None) -> str:
    return _tool_choice_for_main_loop_fn(self._run_policy, round_idx, run_request)


def _unavailable_repl_tool_names(self, tool_calls: list[Any]) -> list[str]:
    """Unavailable tool names in REPL bash-file-change mode."""
    if bool(getattr(self, "_include_write_edit_tools", True)):
        return []
    tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
    available = set(tool_map)
    unavailable: list[str] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        if name and name not in available:
            unavailable.append(str(name))
    return unavailable


@staticmethod
def _tc_function_name(tc: Any) -> str | None:
    return tool_call_function_name(tc, allow_mapping=False)


@staticmethod
def _tc_arguments_dict(tc: Any) -> dict[str, Any]:
    return tool_call_arguments_dict(tc, allow_mapping=False)


@staticmethod
def _repl_file_change_command_from_tool(name: str, args: dict[str, Any]) -> str | None:
    path = str(args.get("path") or "").strip()
    if not path:
        return None
    if name == "edit":
        old = args.get("old_str")
        new = args.get("new_str")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"path = Path({path!r})\n"
            f"old = {old!r}\n"
            f"new = {new!r}\n"
            "text = path.read_text()\n"
            "count = text.count(old)\n"
            "if count != 1:\n"
            "    raise SystemExit(f'exact anchor matched {count} times in {path}')\n"
            "path.write_text(text.replace(old, new, 1))\n"
            "PY"
        )
    if name == "write":
        content = args.get("content")
        if not isinstance(content, str):
            return None
        return (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"Path({path!r}).write_text({content!r})\n"
            "PY"
        )
    return None


def _normalize_repl_file_change_tool_calls(self, tool_calls: list[Any]) -> list[Any]:
    """Convert file-change shaped calls to bash in REPL bash-file-change mode."""
    if bool(getattr(self, "_include_write_edit_tools", True)) or not tool_calls:
        return tool_calls
    tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
    if "bash" not in tool_map:
        return tool_calls
    normalized: list[Any] = []
    changed = 0
    for tc in tool_calls:
        name = self._tc_function_name(tc)
        if name not in {"write", "edit"}:
            normalized.append(tc)
            continue
        command = self._repl_file_change_command_from_tool(
            name,
            self._tc_arguments_dict(tc),
        )
        if not command:
            normalized.append(tc)
            continue
        normalized.append(
            SimpleNamespace(
                id=getattr(tc, "id", ""),
                type=getattr(tc, "type", "function"),
                function=SimpleNamespace(
                    name="bash",
                    arguments=json.dumps(
                        {
                            "command": command,
                            "thought": "Apply the requested file change through bash.",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
        )
        changed += 1
    if changed:
        self._log_info(
            "[tool-calls] normalized %d REPL file-change call(s) to bash",
            changed,
        )
    return normalized


def _block_unavailable_repl_tool_calls(
    self,
    tool_calls: list[Any],
    unavailable_names: list[str],
) -> None:
    """Ask the model to retry with available REPL tools without storing bad args."""
    tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
    available = ", ".join(f"`{name}`" for name in sorted(tool_map))
    self.memory.add_message(
        Message.user_message(
            "[Guard] Unavailable tool call blocked. "
            f"Available tools: {available}. "
            "Use `bash` for file changes; put the complete content or exact "
            "rewrite operation in the bash command. Retry now with one "
            "available tool call.",
        ),
    )
    self._log_info(
        "[tool-calls] unavailable REPL tool blocked; args omitted; count=%d",
        len(unavailable_names or _tool_call_names_from_list(tool_calls)),
    )
