"""Construction of one run-loop control session."""

from __future__ import annotations

import logging
from typing import Any

from inquirycraft.runtime import AgentState

from scienceflow.research.control.agent_session_state import RunSessionState
from scienceflow.runtime.safety.policy.execution_policy import clear_embedded_full_run_result


logger = logging.getLogger("scienceflow")


def initialize_run_session(
    agent: Any,
    *,
    runtime_manages_lifecycle: bool,
) -> tuple[RunSessionState, Any | None]:
    """Reset run-scoped counters and return the sole control-state owner."""
    if not runtime_manages_lifecycle:
        agent.state = AgentState.RUNNING
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_route_tokens_in = 0
    agent._run_route_tokens_out = 0
    agent._run_route_tokens_cached = 0
    agent._run_route_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    agent._call_seq = 0
    agent._run_policy.reset_for_new_run()
    agent._embedded_full_run_done = False
    clear_embedded_full_run_result(agent._workspace_dir)
    agent._run_control_user_injections = 0
    agent._lnr_snapshot_ok = False
    agent._lnr_snapshot_reason = ""
    agent._lnr_last_valid_solution_sha = ""
    agent._lnr_last_valid_submission_sha = ""
    agent._lnr_last_valid_bash_cmd = ""
    agent._lnr_result_md_after_success_pending = False
    agent._lnr_ledger_committed = False
    agent._lnr_ledger_step_count = 0
    agent._lnr_ledger_new_entries = 0
    if str(getattr(agent, "_lnr_agentic_text_only_route_prompt", "") or "").strip():
        agent._lnr_agentic_text_only_route_prompt_injected = False
        agent._lnr_agentic_text_only_route_without_result_md = False
        try:
            (agent._agentic_route_log_dir() / "agentic_route_response.md").unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug(
                "[agentic-route] stale route response cleanup skipped",
                exc_info=True,
            )
    agent._consecutive_write_syntax_fails = 0
    agent._initial_solution_sha = agent._sha256_of_solution()
    agent._mid_run_compacted = False
    guard_manager = getattr(agent, "_guard_manager", None)
    if guard_manager is not None:
        guard_manager.reset_all()
    if hasattr(agent, "_reset_edit_read_guard_state"):
        agent._reset_edit_read_guard_state()
    session = RunSessionState.start(agent.max_steps)
    agent._effective_max_steps = session.effective_max_steps
    return session, guard_manager


__all__ = ("initialize_run_session",)
