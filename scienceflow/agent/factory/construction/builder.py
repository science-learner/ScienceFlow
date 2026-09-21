"""Small typed orchestrator for ScienceAgent construction phases."""

from __future__ import annotations

from typing import Any

from scienceflow.agent.factory.construction.assembly import (
    AgentConstructionOptions,
    _AgentConstructionState,
    _initialize_agent_base,
    _initialize_agent_guards,
    _initialize_agent_memory,
    _initialize_agent_prompt_and_logs,
    _initialize_agent_state,
)


class ScienceAgentBuilder:
    """Apply construction phases to one host from one typed option object."""

    def __init__(self, host: Any, options: AgentConstructionOptions) -> None:
        self.host = host
        self.options = options
        self.state = _AgentConstructionState()

    def build(self) -> None:
        _initialize_agent_base(self.host, self.options, self.state)
        _initialize_agent_memory(self.host, self.options, self.state)
        _initialize_agent_prompt_and_logs(self.host, self.options, self.state)
        _initialize_agent_state(self.host, self.options, self.state)
        _initialize_agent_guards(self.host, self.options, self.state)


__all__ = ["AgentConstructionOptions", "ScienceAgentBuilder"]
