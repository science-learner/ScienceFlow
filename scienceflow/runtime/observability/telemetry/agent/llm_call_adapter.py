"""ScienceFlow adapter from agent-call observations to the workflow time trace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from scienceflow.foundation.config.schema.settings import Config


def _model_config_paths(cfg: Config) -> tuple[str, ...]:
    agent = getattr(cfg, "agent", None)
    values = (
        getattr(getattr(agent, "code", None), "model_config_path", ""),
        getattr(getattr(agent, "feedback", None), "model_config_path", ""),
    )
    return tuple(dict.fromkeys(str(value) for value in values if value))


def make_llm_call_tracer(
    cfg: Config,
    *,
    node_id: str = "repl",
    process_id: int | str | None = None,
    fork_class: str | None = None,
    detail_prefix: str = "mode=repl",
) -> Callable[[dict[str, Any]], None] | None:
    """Build an ``on_llm_call`` hook that writes rows to ``scienceflow_time_trace.csv``."""
    if not bool(getattr(cfg, "enable_time_trace", True)):
        return None
    from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
        load_model_config_prices,
    )
    from scienceflow.runtime.observability.telemetry.trace.time_trace import (
        TimeTracer,
        merge_trace_details,
    )

    tracer = TimeTracer(
        Path(cfg.log_dir),
        enabled=True,
        price_table_config=load_model_config_prices(*_model_config_paths(cfg)),
    )

    def _hook(payload: dict[str, Any]) -> None:
        op = str(payload.get("operation", "llm"))
        dur = float(payload.get("duration_sec", 0.0) or 0.0)
        status = str(payload.get("status", "ok"))
        seq = payload.get("call_seq", "")
        rnd = payload.get("round_idx")
        rec = bool(payload.get("recovery", False))
        model = (payload.get("model") or "") or ""
        turn_kind = payload.get("turn_kind")
        first_tool = payload.get("first_tool_name")
        first_bash_kind = payload.get("first_bash_kind")
        detail = merge_trace_details(
            f"model={str(model)[:120]}" if model else "",
            detail_prefix,
            f"llm_role={str(payload.get('llm_role') or 'code')[:40]}",
            f"call_seq={seq}",
            f"round={rnd}" if rnd is not None else "round=-",
            f"recovery={int(rec)}",
            f"turn={turn_kind}" if isinstance(turn_kind, str) and turn_kind.strip() else "",
            f"tool={str(first_tool)[:40]}"
            if isinstance(first_tool, str) and first_tool.strip()
            else "",
            f"bash_kind={str(first_bash_kind)[:40]}"
            if isinstance(first_bash_kind, str) and first_bash_kind.strip()
            else "",
            max_len=200,
        )
        tracer.record(
            "llm_api",
            op,
            dur,
            status=status,
            node_id=node_id,
            process_id=str(process_id) if process_id is not None else "",
            tokens_input=payload.get("tokens_input"),
            tokens_output=payload.get("tokens_output"),
            tokens_cached=payload.get("tokens_cached"),
            ttft_sec=payload.get("ttft_sec"),
            tpot_ms=payload.get("tpot_ms"),
            pool_index=payload.get("pool_index"),
            failover_count=payload.get("failover_count"),
            detail=detail,
            fork_class=fork_class,
        )

    return _hook

__all__ = ["make_llm_call_tracer"]
