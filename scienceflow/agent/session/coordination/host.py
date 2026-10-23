"""ScienceAgent host entry point and session factory."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from scienceflow.agent.session.coordination.contracts import AgentSessionSpec
from scienceflow.agent.session.execution.runtime import ScienceFlowAgentSession


class ScienceFlowAgentSessionFactory:
    def build(self, host: Any, spec: AgentSessionSpec) -> ScienceFlowAgentSession:
        return ScienceFlowAgentSession(host, spec)


async def run_science_agent(
    host: Any,
    request: str | None = None,
    *,
    first_round_tool_choice: str | None = None,
) -> str:
    """Run one ScienceFlow session through InquiryCraft's sole AgentRuntime."""
    workspace = host._workspace_dir
    worker_id = str(getattr(host, "_scienceflow_worker_id", "") or "W00")
    run_id = str(
        getattr(host, "_scienceflow_run_id", "")
        or os.environ.get("SCIENCEFLOW_RUN_ID", "")
        or workspace.parent.name
        or "scienceflow"
    )
    session_id = f"{run_id}-{worker_id}-{uuid4().hex[:12]}"
    session = host._agent_session_factory.build(
        host,
        AgentSessionSpec(
            request=request,
            workspace=workspace,
            model=str(getattr(host.llm, "model", "") or "scienceflow"),
            max_turns=int(host.max_steps),
            first_round_tool_choice=first_round_tool_choice,
            run_id=run_id,
            session_id=session_id,
            worker_id=worker_id,
            stage_id=str(getattr(host, "_scienceflow_stage_id", "") or "draft"),
            metadata={
                "lineage_id": str(getattr(host, "_scienceflow_lineage_id", "") or ""),
                "node_uid": str(getattr(host, "_scienceflow_node_uid", "") or ""),
            },
        ),
    )
    return await session.run()


__all__ = ["ScienceFlowAgentSessionFactory", "run_science_agent"]
