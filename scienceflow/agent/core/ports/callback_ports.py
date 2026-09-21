# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Typed callback capabilities injected at the ScienceAgent composition edge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol


class CandidateArchivePort(Protocol):
    def __call__(
        self,
        *,
        agent: Any,
        tool_name: str,
        args: dict[str, Any],
        tool_result: Any,
    ) -> Any: ...


class StageCapturePort(Protocol):
    def __call__(
        self,
        *,
        agent: Any,
        args: dict[str, Any],
        tool_result: Any,
    ) -> Any: ...


class MetricInterpretationPort(Protocol):
    def __call__(
        self,
        *,
        agent: Any,
        stdout: str,
        script_label: str,
    ) -> Any: ...


class TextOnlyDecisionPort(Protocol):
    def __call__(
        self,
        *,
        agent: Any,
        assistant_text: str,
        round_idx: int,
        max_steps: int,
    ) -> Any: ...


class ContextLimitEstraPort(Protocol):
    def __call__(
        self,
        *,
        agent: Any,
        round_idx: int,
        max_steps: int,
        omitted: int,
    ) -> Any: ...


class ContextCompactObserver(Protocol):
    def __call__(self, *, phase: str, **payload: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class AgentCallbackPorts:
    """Narrow callback capabilities; absent capabilities are explicitly ``None``."""

    candidate_archive: CandidateArchivePort | None = None
    stage_capture: StageCapturePort | None = None
    metric_interpretation: MetricInterpretationPort | None = None
    text_only_decision: TextOnlyDecisionPort | None = None
    context_limit_estra: ContextLimitEstraPort | None = None
    context_compact_observer: ContextCompactObserver | None = None

    def with_overrides(self, **updates: Any) -> AgentCallbackPorts:
        return replace(self, **updates)


_LEGACY_CALLBACK_ATTRIBUTES = {
    "candidate_archive": "_lnr_candidate_archive_callback",
    "stage_capture": "_lnr_stage_capture_callback",
    "metric_interpretation": "_lnr_metric_output_interpretation_callback",
    "text_only_decision": "_lnr_text_only_callback",
    "context_limit_estra": "_context_limit_estra_callback",
    "context_compact_observer": "_context_compact_event_callback",
}


def callback_ports_for(agent: Any) -> AgentCallbackPorts:
    ports = getattr(agent, "callback_ports", None)
    return ports if isinstance(ports, AgentCallbackPorts) else AgentCallbackPorts()


def install_callback_ports(agent: Any, ports: AgentCallbackPorts) -> None:
    installer = getattr(agent, "configure_callback_ports", None)
    if callable(installer):
        installer(ports)
        return
    agent.callback_ports = ports


def resolve_agent_callback(agent: Any, capability: str) -> Any | None:
    """Resolve a typed port, with a read-only legacy fallback during migration."""

    ports = callback_ports_for(agent)
    callback = getattr(ports, capability)
    if callable(callback):
        return callback
    legacy_name = _LEGACY_CALLBACK_ATTRIBUTES[capability]
    legacy_callback = getattr(agent, legacy_name, None)
    return legacy_callback if callable(legacy_callback) else None


__all__ = [
    "AgentCallbackPorts",
    "CandidateArchivePort",
    "ContextCompactObserver",
    "ContextLimitEstraPort",
    "MetricInterpretationPort",
    "StageCapturePort",
    "TextOnlyDecisionPort",
    "callback_ports_for",
    "install_callback_ports",
    "resolve_agent_callback",
]
