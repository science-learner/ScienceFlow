"""Shared persistence for every ScienceFlow provider call."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from inquirycraft.runtime import ProviderObservation, RuntimeContext

logger = logging.getLogger("scienceflow")


def runtime_log_dir(host: Any, workspace: Path) -> Path:
    """Resolve the durable audit root used by regular and ephemeral sessions."""
    override = getattr(host, "_scienceflow_runtime_log_dir", None)
    path = (
        Path(override).expanduser().resolve(strict=False)
        if override is not None and str(override).strip()
        else workspace / ".logs"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


class ScienceFlowProviderAudit:
    """Project InquiryCraft provider observations to ScienceFlow telemetry."""

    def __init__(
        self,
        host: Any | None,
        log_dir: Path,
        *,
        llm_override: Any | None = None,
        llm_role: str | None = None,
    ) -> None:
        self.host = host
        self.log_dir = log_dir
        self.llm_override = llm_override
        self.llm_role = llm_role
        self.call_seq = 0

    def observe(
        self, observation: ProviderObservation, context: RuntimeContext
    ) -> None:
        if observation.phase == "completed" and self.host is not None:
            self.host._run_tokens_in += observation.usage.input_tokens
            self.host._run_tokens_out += observation.usage.output_tokens
            self.host._run_tokens_cached += observation.usage.cached_input_tokens
            self.host._run_llm_calls += 1
            if observation.metadata.get("call_kind") == "ephemeral":
                prefix = (
                    "_run_compact"
                    if str(observation.metadata.get("turn_kind") or "").startswith(
                        "compact"
                    )
                    else "_run_route"
                )
                setattr(
                    self.host,
                    f"{prefix}_tokens_in",
                    int(getattr(self.host, f"{prefix}_tokens_in", 0) or 0)
                    + observation.usage.input_tokens,
                )
                setattr(
                    self.host,
                    f"{prefix}_tokens_out",
                    int(getattr(self.host, f"{prefix}_tokens_out", 0) or 0)
                    + observation.usage.output_tokens,
                )
                setattr(
                    self.host,
                    f"{prefix}_tokens_cached",
                    int(getattr(self.host, f"{prefix}_tokens_cached", 0) or 0)
                    + observation.usage.cached_input_tokens,
                )
                setattr(
                    self.host,
                    f"{prefix}_llm_calls",
                    int(getattr(self.host, f"{prefix}_llm_calls", 0) or 0) + 1,
                )
        if observation.phase not in {"completed", "failed"}:
            return
        self.call_seq += 1
        turn_kind = str(
            observation.metadata.get("turn_kind")
            or ("tool" if observation.tool_call_count else "qa")
        )
        extra = (
            {
                "turn_kind": turn_kind,
                **(
                    {"first_tool_name": observation.tool_names[0]}
                    if observation.tool_names
                    else {}
                ),
            }
            if observation.phase == "completed"
            else {}
        )
        completed = observation.phase == "completed"
        record = getattr(self.host, "_record_llm_call", None)
        if callable(record):
            record(
                "ask_tool_stream",
                observation.elapsed_sec,
                max(0, context.turn_id - 1),
                "ok" if completed else "error",
                recovery=False,
                tokens_input=observation.usage.input_tokens if completed else None,
                tokens_output=observation.usage.output_tokens if completed else None,
                tokens_cached=(
                    observation.usage.cached_input_tokens if completed else None
                ),
                ttft_sec=observation.ttft_sec,
                tpot_ms=(
                    observation.tpot_sec * 1000
                    if observation.tpot_sec is not None
                    else None
                ),
                llm_override=self.llm_override,
                llm_role=self.llm_role,
                **extra,
            )
        self._append(observation, context)

    def _append(
        self, observation: ProviderObservation, context: RuntimeContext
    ) -> None:
        usage = observation.usage
        correlation = dict(observation.metadata.get("correlation") or {})
        turn_kind = str(
            observation.metadata.get("turn_kind")
            or ("tool" if observation.tool_call_count else "qa")
        )
        usage_known = bool(
            usage.input_tokens or usage.output_tokens or usage.cached_input_tokens
        )
        row = {
            "schema_version": 1,
            "timestamp": time.time() - max(0, observation.elapsed_sec or 0),
            "call_seq": self.call_seq,
            "phase": observation.phase,
            "session_id": context.session_id,
            "run_id": context.run_id,
            "worker_id": context.agent_id,
            "stage_id": correlation.get("stage_id")
            or observation.metadata.get("stage_id")
            or context.metadata.get("stage_id", "draft"),
            "lineage_id": correlation.get("lineage_id")
            or observation.metadata.get("lineage_id"),
            "node_uid": correlation.get("node_uid")
            or observation.metadata.get("node_uid"),
            "call_id": correlation.get("call_id")
            or observation.metadata.get("call_id"),
            "turn_id": context.turn_id,
            "model": observation.model,
            "llm_role": self.llm_role,
            "attempt": observation.attempt,
            "elapsed_sec": observation.elapsed_sec,
            "ttft_sec": observation.ttft_sec,
            "tpot_sec": observation.tpot_sec,
            "tokens_input": usage.input_tokens if usage_known else None,
            "tokens_output": usage.output_tokens if usage_known else None,
            "tokens_cached": usage.cached_input_tokens if usage_known else None,
            "cache_rate": (
                usage.cached_input_tokens / usage.input_tokens
                if usage_known and usage.input_tokens
                else None
            ),
            "usage_status": "known" if usage_known else "unknown",
            "finish_reason": observation.finish_reason,
            "tool_names": list(observation.tool_names),
            "call_kind": observation.metadata.get("call_kind", "agent"),
            "turn_kind": turn_kind,
            "stable_prefix_hash": observation.metadata.get("stable_prefix_hash"),
            "provider_context_hash": observation.metadata.get("provider_context_hash"),
            "provider_message_count": observation.metadata.get(
                "provider_message_count"
            ),
            "compaction_generation": observation.metadata.get(
                "compaction_generation", 0
            ),
            "route": observation.metadata.get("route"),
            "correlation": correlation,
        }
        path = self.log_dir / "agent_provider_calls.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        except OSError as exc:
            warning = getattr(self.host, "_log_warning", None)
            if callable(warning):
                warning("[telemetry] provider audit append failed: %s", exc)
            else:
                logger.warning("[telemetry] provider audit append failed: %s", exc)


__all__ = ["ScienceFlowProviderAudit", "runtime_log_dir"]
