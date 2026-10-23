"""Role-aware ScienceAgent construction boundary."""

from scienceflow.agent.factory.contracts import AgentBuildRecord, AgentBuildRequest
from scienceflow.agent.factory.composition import create_science_agent
from scienceflow.agent.factory.service import AgentFactory

__all__ = [
    "AgentBuildRecord",
    "AgentBuildRequest",
    "AgentFactory",
    "create_science_agent",
]
