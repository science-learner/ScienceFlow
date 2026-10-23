"""Central role-aware agent construction and soft telemetry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Callable, Mapping

from scienceflow.agent.factory.contracts import AgentBuildRecord, AgentBuildRequest


AgentCreator = Callable[..., Any]
AgentBuildTraceSink = Callable[[dict[str, Any]], None]


class AgentFactory:
    def __init__(
        self,
        creator: AgentCreator,
        *,
        trace_sink: AgentBuildTraceSink | None = None,
    ) -> None:
        self.creator = creator
        self.trace_sink = trace_sink
        self.records: list[AgentBuildRecord] = []
        self._sequence = 0

    def create(
        self,
        role: str,
        *,
        correlation: Mapping[str, str] | None = None,
        **options: Any,
    ) -> Any:
        request = AgentBuildRequest(
            role=str(role or "science_agent"),
            options=dict(options),
            correlation=dict(correlation or {}),
        )
        self._sequence += 1
        sequence = self._sequence
        option_keys = tuple(sorted(request.options))
        seed = json.dumps(
            {
                "role": request.role,
                "sequence": sequence,
                "option_keys": option_keys,
                "correlation": dict(request.correlation),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        build_id = "agent_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        try:
            agent = self.creator(**dict(request.options))
        except Exception as exc:
            self._record(
                AgentBuildRecord(
                    build_id=build_id,
                    sequence=sequence,
                    role=request.role,
                    option_keys=option_keys,
                    correlation=request.correlation,
                    outcome="failed",
                    error_type=type(exc).__name__,
                )
            )
            raise
        self._record(
            AgentBuildRecord(
                build_id=build_id,
                sequence=sequence,
                role=request.role,
                option_keys=option_keys,
                correlation=request.correlation,
                outcome="created",
            )
        )
        return agent

    def _record(self, record: AgentBuildRecord) -> None:
        self.records.append(record)
        if self.trace_sink is None:
            return
        payload = asdict(record)
        payload["schema_version"] = record.contract_schema_version()
        try:
            self.trace_sink(payload)
        except Exception:
            # Construction telemetry is deliberately soft.
            pass


__all__ = ["AgentFactory"]
