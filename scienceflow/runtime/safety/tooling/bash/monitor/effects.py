"""Termination, cleanup, and lease effects for monitored Bash processes."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _executed_resource_termination_feedback,
    _kill_revalidation_allows_termination,
    _resource_call,
)
from scienceflow.runtime.safety.tooling.bash.core.shared import (
    _cleanup_workspace_gpu_processes,
    logger,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.utilization import (
    process_tree_cpu_snapshot as _process_tree_cpu_snapshot,
)


class ResourceTerminationEffects:
    """Apply process, workspace, and lease mutations chosen by guard decisions."""

    def _recoverable_stop_marker_seen(self) -> bool:
        if self.run_state_path is None:
            return False
        try:
            stat = self.run_state_path.stat()
        except OSError:
            return False
        if stat.st_mtime + 1.0 < self.start_wall or stat.st_size <= 0:
            return False
        try:
            state = json.loads(self.run_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(state, dict):
            return False
        return str(state.get("status") or "").strip().lower() in {"interrupted", "completed"}

    async def _terminate_resource_guard_process(self) -> None:
        if not self.tool.resource_recoverable_stop_enabled:
            await self.session.terminate()
            return
        await self.session.terminate(
            recoverable=True,
            marker_check=self._recoverable_stop_marker_seen,
            sigusr1_grace=float(self.tool.resource_recoverable_stop_sigusr1_grace_sec or 0.0),
            marker_exit_grace=float(self.tool.resource_recoverable_stop_marker_exit_grace_sec or 0.0),
            sigterm_grace=float(self.tool.resource_recoverable_stop_sigterm_grace_sec or 0.0),
        )

    async def _run_workspace_gpu_cleanup(self, *, reason: str, force: bool = False) -> dict[str, Any]:
        if self.workspace_cleanup_done[0] and (not force):
            return {}
        cleanup_gpu_ids = self.task_physical_gpu_pool or self.gpu_ids
        if not cleanup_gpu_ids or self.tool.resource_observer is None:
            return {}
        self.workspace_cleanup_done[0] = True
        try:
            cleanup = await asyncio.to_thread(_cleanup_workspace_gpu_processes, workspace_dir=self.ws, allowed_gpu_ids=cleanup_gpu_ids, task_started_at=self.start_wall, reason=reason, dry_run=False, sigterm_grace_sec=0.5)
        except Exception:
            logger.debug("[bash-resource] workspace gpu cleanup failed", exc_info=True)
            return {}
        if not isinstance(cleanup, dict):
            return {}
        killed_count = int(cleanup.get("killed_count") or 0)
        violation_count = int(cleanup.get("violation_count") or 0)
        skipped = cleanup.get("skipped") if isinstance(cleanup.get("skipped"), list) else []
        interesting_skips = [row for row in skipped if str((row or {}).get("skip_reason") or "") not in {"pid_disappeared", "workspace_mismatch", "process_started_before_task"}]
        if killed_count > 0:
            action = "gpu_orphan_cleanup"
        elif violation_count > 0:
            action = "stop_boundary_violation"
        elif interesting_skips:
            _resource_call(self.tool.resource_observer, "resource_cleanup_heartbeat", self.resource_job_id, action="gpu_cleanup_skipped", reason=reason, elapsed_sec=max(0.0, time.time() - self.start), placement=cleanup)
            return cleanup
        else:
            return cleanup
        _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action=action, reason=reason, elapsed_sec=max(0.0, time.time() - self.start), placement=cleanup)
        return cleanup

    def _grant_shared_gpu(self, decision: dict, arbiter: dict, action: str, elapsed_now: float, reason: str, feedback: str) -> bool:
        grant = _resource_call(self.tool.resource_observer, "apply_task_gpu_share_decision", self.resource_job_id, arbiter_action=action, proposal=decision.get("proposal"), arbiter_decision=arbiter.get("decision"), elapsed_sec=elapsed_now)
        _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="grant_shared_gpu_lease" if isinstance(grant, dict) and grant.get("granted") else "grant_shared_gpu_lease_denied", reason=reason, elapsed_sec=elapsed_now)
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            state = "granted shared GPU lease" if isinstance(grant, dict) and grant.get("granted") else "could not grant shared GPU lease"
            self.on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: {reason}\n")
            grant_feedback = str((grant or {}).get("feedback") or feedback).strip() if isinstance(grant, dict) else feedback
            if grant_feedback:
                self.on_output(grant_feedback.rstrip() + "\n")
        self.guard_observe_more_until[0] = time.time() + 30.0
        return False

    def _release_idle_lease(self, arbiter: dict, elapsed_now: float, reason: str, feedback: str) -> bool:
        release = _resource_call(self.tool.resource_observer, "release_idle_gpu_lease", self.resource_job_id, elapsed_sec=elapsed_now, reason=reason, feedback=feedback)
        _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="release_idle_lease" if isinstance(release, dict) and release.get("released") else "release_idle_lease_denied", reason=reason, elapsed_sec=elapsed_now)
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            state = "released idle GPU lease" if isinstance(release, dict) and release.get("released") else "could not release idle GPU lease"
            self.on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: {reason}\n")
            if feedback:
                self.on_output(feedback.rstrip() + "\n")
        self.guard_observe_more_until[0] = time.time() + 300.0
        return False

    async def _execute_arbiter_termination(self, arbiter: dict, action: str, elapsed_now: float, stdout_age: float, reason: str, feedback: str) -> bool:
        self.guard_reason[0] = reason
        guard_action = {"STOP_BOUNDARY_VIOLATION": "stop_boundary_violation"}.get(action, "terminate_by_arbiter")
        decision_payload = arbiter.get("decision") if isinstance(arbiter.get("decision"), dict) else {}
        gate_payload = decision_payload.get("gate") if isinstance(decision_payload.get("gate"), dict) else {}
        if action == "KILL_AND_REPLAN":
            self._maybe_emit_artifact_progress(elapsed_now, force=True)
            fresh_process_tree_cpu = _process_tree_cpu_snapshot(getattr(self.proc, "pid", None))
            revalidation = _resource_call(
                self.tool.resource_observer,
                "revalidate_kill_intent",
                self.resource_job_id,
                arbiter_decision=decision_payload,
                elapsed_sec=elapsed_now,
                stdout_age_sec=stdout_age,
                stdout_lines=int(self.stream_stats["stdout_lines"]),
                stdout_bytes=int(self.stream_stats["stdout_bytes"]),
                metric_history_text=str(self.stream_stats.get("metric_history_text") or ""),
                metric_history_line_count=len(self.stream_stats.get("metric_history_lines") or []),
                saw_training_progress=bool(self.stream_stats["saw_training_progress"]),
                saw_final_score=bool(self.stream_stats["saw_final_score"]),
                current_phase=str(self.stream_stats.get("current_phase") or ""),
                invalid_metric_events=int(self.stream_stats.get("invalid_metric_events") or 0),
                zero_score_events=int(self.stream_stats.get("zero_score_events") or 0),
                last_invalid_metric_text=str(self.stream_stats.get("last_invalid_metric_text") or ""),
                last_zero_score_text=str(self.stream_stats.get("last_zero_score_text") or ""),
                terminal_signal_events=int(self.stream_stats.get("terminal_signal_events") or 0),
                terminal_signal_kind=str(self.stream_stats.get("terminal_signal_kind") or ""),
                last_terminal_signal_text=str(self.stream_stats.get("last_terminal_signal_text") or ""),
                process_tree_cpu=fresh_process_tree_cpu,
                **self.tool._deadline_state(),
            )
            hard_safety = str(gate_payload.get("kill_class") or "") == "hard_safety"
            if not _kill_revalidation_allows_termination(revalidation, hard_safety=hard_safety):
                revalidation_payload = revalidation if isinstance(revalidation, dict) else {}
                revalidation_available = bool(revalidation_payload.get("enabled") is True and "allow_kill" in revalidation_payload)
                revalidation_reason = str(revalidation_payload.get("reason") or ("stale_kill_intent" if revalidation_available else "kill_revalidation_unavailable"))
                protective_changes = revalidation_payload.get("protective_changes") or []
                self.guard_reason[0] = ""
                self.guard_observe_more_until[0] = time.time() + 120.0
                _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="stale_kill_intent_denied" if revalidation_available else "kill_revalidation_unavailable", reason=revalidation_reason, elapsed_sec=elapsed_now, protective_changes=protective_changes)
                if self.on_output is not None:
                    if self.heartbeat_active[0]:
                        self.on_output("\r\x1b[K")
                        self.heartbeat_active[0] = False
                    detail = ",".join((str(item) for item in protective_changes))
                    if not detail:
                        detail = revalidation_reason
                    self.on_output(f"[resource arbiter] kill denied by revalidation after {elapsed_now:.0f}s: {detail}\n")
                return False
        await self._terminate_resource_guard_process()
        _resource_call(
            self.tool.resource_observer,
            "resource_guard_action",
            self.resource_job_id,
            action=guard_action,
            reason=reason,
            elapsed_sec=elapsed_now,
            arbiter_decision_id=decision_payload.get("decision_id"),
            proposal_id=decision_payload.get("proposal_id") or arbiter.get("proposal_id"),
            kill_class=gate_payload.get("kill_class"),
            strict_gate_result="allow" if bool(gate_payload.get("allowed")) else "blocked",
            strict_gate_reason=gate_payload.get("blocked_reason"),
            advisory_support=gate_payload.get("advisory_support"),
            captured_advisory=gate_payload.get("captured_advisory"),
        )
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            verb = "KILL EXECUTED; command process has terminated"
            if action == "STOP_BOUNDARY_VIOLATION":
                verb = "STOP EXECUTED; command process has terminated"
            self.on_output(f"[resource arbiter] {verb} after {elapsed_now:.0f}s: {reason}\n")
            executed_feedback = _executed_resource_termination_feedback(feedback, action=action)
            self.on_output(executed_feedback.rstrip() + "\n")
        return True




__all__ = ["ResourceTerminationEffects"]
