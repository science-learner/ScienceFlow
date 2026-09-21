"""Resource guard decisions, arbiter routing, and watchdog recovery."""

from __future__ import annotations

import asyncio
import time

from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _resource_async_call,
    _resource_call,
)
from scienceflow.runtime.safety.tooling.bash.core.shared import (
    _build_gpu_boundary_feedback,
    _process_tree_gpu_placement_snapshot,
    logger,
    terminate_tree,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.utilization import (
    process_tree_cpu_snapshot as _process_tree_cpu_snapshot,
)


class ResourceGuardMonitor:
    """Own resource decisions and watchdog state without process effects."""

    async def _apply_arbiter_continuation(self, decision: dict, arbiter: dict, action: str, elapsed_now: float, stdout_age: float, reason: str, feedback: str) -> bool:
        if action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "CONTINUE", "DENY_KILL", "CONTINUE_SHARED_OBSERVE", "DENY_SHARE_USE_CPU_SUPPORT"}:
            if action in {"DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
                _resource_call(self.tool.resource_observer, "apply_task_gpu_share_decision", self.resource_job_id, arbiter_action=action, proposal=decision.get("proposal"), arbiter_decision=arbiter.get("decision"), elapsed_sec=elapsed_now)
            arbiter_decision = arbiter.get("decision", {}) if isinstance(arbiter.get("decision"), dict) else {}
            execution_outcome = str(arbiter.get("execution_outcome") or arbiter_decision.get("execution_outcome") or "").upper()
            if execution_outcome == "TIMEBOX":
                ttl = arbiter.get("computed_timebox_sec") or arbiter_decision.get("computed_timebox_sec")
            elif action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL", "CONTINUE_SHARED_OBSERVE"}:
                ttl = arbiter_decision.get("observe_more_sec")
            else:
                ttl = arbiter_decision.get("ttl_sec")
            observe_more_sec = float(ttl or 300.0)
            self.guard_observe_more_until[0] = time.time() + max(1.0, observe_more_sec)
        else:
            self.guard_recommendation_emitted[0] = True
        arbiter_decision = arbiter.get("decision", {}) if isinstance(arbiter.get("decision"), dict) else {}
        execution_outcome = str(arbiter.get("execution_outcome") or arbiter_decision.get("execution_outcome") or "").upper()
        action_name = (
            "arbiter_timebox"
            if execution_outcome == "TIMEBOX"
            else {
                "OBSERVE_MORE": "arbiter_observe_more",
                "MARK_STALLED_NO_KILL": "arbiter_mark_stalled_no_kill",
                "CONTINUE": "arbiter_continue",
                "DENY_KILL": "arbiter_deny_kill",
                "CONTINUE_SHARED_OBSERVE": "arbiter_continue_shared_observe",
                "DENY_SHARE_USE_CPU_SUPPORT": "arbiter_deny_share_use_cpu_support",
            }.get(action, "arbiter_no_action")
        )
        _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action=action_name, reason=reason, elapsed_sec=elapsed_now)
        suppress_feedback = bool(arbiter.get("suppress_main_agent_feedback"))
        if self.on_output is not None and (not suppress_feedback):
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            self.on_output(f"[resource arbiter] {action.lower()} after {elapsed_now:.0f}s: {reason}\n")
            if feedback:
                self.on_output(feedback.rstrip() + "\n")
        if action in {"CONTINUE", "DENY_KILL", "OBSERVE_MORE", "MARK_STALLED_NO_KILL"}:
            post_share = _resource_call(
                self.tool.resource_observer,
                "post_arbiter_continue_review",
                self.resource_job_id,
                arbiter_action=action,
                inferred_class=self.effective_resource_class,
                gpu_ids=self.gpu_ids,
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
                process_tree_cpu=_process_tree_cpu_snapshot(getattr(self.proc, "pid", None)),
                **self.tool._deadline_state(),
            )
            if isinstance(post_share, dict) and post_share.get("arbiter_review") and post_share.get("arbiter_enabled"):
                share_arbiter = await _resource_async_call(self.tool.resource_observer, "arbiter_decide", self.resource_job_id, proposal=post_share.get("proposal"), decision_preview=post_share)
                if isinstance(share_arbiter, dict) and share_arbiter.get("enabled"):
                    share_action = str(share_arbiter.get("action") or "").upper()
                    if share_action in {"GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
                        share_apply = _resource_call(self.tool.resource_observer, "apply_task_gpu_share_decision", self.resource_job_id, arbiter_action=share_action, proposal=post_share.get("proposal"), arbiter_decision=share_arbiter.get("decision"), elapsed_sec=elapsed_now)
                        if self.on_output is not None:
                            if self.heartbeat_active[0]:
                                self.on_output("\r\x1b[K")
                                self.heartbeat_active[0] = False
                            state = "granted shared GPU lease" if isinstance(share_apply, dict) and share_apply.get("granted") else share_action.lower()
                            self.on_output(f"[resource arbiter] {state} after {elapsed_now:.0f}s: task_gpu_share_review\n")
                            share_feedback = str((share_apply or {}).get("feedback") or share_arbiter.get("feedback") or "").strip() if isinstance(share_apply, dict) else str(share_arbiter.get("feedback") or "").strip()
                            if share_feedback:
                                self.on_output(share_feedback.rstrip() + "\n")
        return False

    async def _handle_arbiter_review(self, decision: dict, elapsed_now: float, stdout_age: float, reason: str, feedback: str) -> bool | None:
        arbiter = await _resource_async_call(self.tool.resource_observer, "arbiter_decide", self.resource_job_id, proposal=decision.get("proposal"), decision_preview=decision)
        if isinstance(arbiter, dict) and arbiter.get("enabled"):
            action = str(arbiter.get("action") or "").upper()
            arbiter_feedback = str(arbiter.get("feedback") or "").strip()
            if arbiter_feedback:
                feedback = arbiter_feedback
                self.guard_feedback[0] = feedback
            if action == "GRANT_SHARED_GPU_LEASE":
                return self._grant_shared_gpu(decision, arbiter, action, elapsed_now, reason, feedback)
            if action == "RELEASE_IDLE_LEASE":
                return self._release_idle_lease(arbiter, elapsed_now, reason, feedback)
            if action in {"KILL_AND_REPLAN", "STOP_BOUNDARY_VIOLATION"}:
                return await self._execute_arbiter_termination(arbiter, action, elapsed_now, stdout_age, reason, feedback)
            return await self._apply_arbiter_continuation(decision, arbiter, action, elapsed_now, stdout_age, reason, feedback)
        return None

    async def _handle_resource_review(self, decision: dict, elapsed_now: float, stdout_age: float) -> bool:
        review_only = bool(decision.get("arbiter_review")) and (not bool(decision.get("would_terminate")))
        if self.guard_observe_more_until[0] and time.time() < self.guard_observe_more_until[0]:
            return False
        if self.guard_observe_more_until[0] and time.time() >= self.guard_observe_more_until[0]:
            self.guard_observe_more_until[0] = 0.0
        internal_arbiter_review = bool(decision.get("arbiter_enabled") and decision.get("requires_llm_decision"))
        if self.guard_recommendation_emitted[0] and (not review_only) and (not internal_arbiter_review):
            return False
        reason = str(decision.get("reason") or "resource_guard_recommendation")
        feedback = str(decision.get("feedback") or "").strip()
        arbiter_result = await self._handle_arbiter_review(decision, elapsed_now, stdout_age, reason, feedback)
        if arbiter_result is not None:
            return arbiter_result
        self.guard_recommendation_emitted[0] = True
        if feedback:
            self.guard_feedback[0] = feedback
        _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="recommend_stop", reason=reason, elapsed_sec=elapsed_now)
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            self.on_output(f"[resource guard] recommends LLM review after {elapsed_now:.0f}s: {reason}; command not terminated\n")
            if feedback:
                self.on_output(feedback.rstrip() + "\n")
        return False

    def _emit_observe_first_health(self, elapsed_now: float, stdout_age: float) -> None:
        if (
            not self.observe_first_active
            or self.observe_first_window_sec <= 0
            or self.observe_first_health_emitted[0]
            or elapsed_now < self.observe_first_window_sec
        ):
            return
        self.observe_first_health_emitted[0] = True
        _resource_call(
            self.tool.resource_observer,
            "observe_first_healthy_continue",
            self.resource_job_id,
            elapsed_sec=elapsed_now,
            stdout_age_sec=stdout_age,
            stdout_lines=int(self.stream_stats["stdout_lines"]),
            stdout_bytes=int(self.stream_stats["stdout_bytes"]),
            metric_history_text=str(
                self.stream_stats.get("metric_history_text") or ""
            ),
            metric_history_line_count=len(
                self.stream_stats.get("metric_history_lines") or []
            ),
            saw_training_progress=bool(self.stream_stats["saw_training_progress"]),
            saw_final_score=bool(self.stream_stats["saw_final_score"]),
            current_phase=str(self.stream_stats.get("current_phase") or ""),
        )

    async def _stop_gpu_boundary_violation(self, elapsed_now: float) -> bool:
        placement_interval = 5.0 if elapsed_now < 120.0 else 15.0
        if (
            not self.gpu_ids
            or time.monotonic() - self.placement_audit_last[0] < placement_interval
        ):
            return False
        self.placement_audit_last[0] = time.monotonic()
        placement = _process_tree_gpu_placement_snapshot(
            getattr(self.proc, "pid", None),
            self.gpu_ids,
        )
        violations = placement.get("violations") if isinstance(placement, dict) else []
        if not violations:
            return False
        reason = "task_gpu_boundary_violation"
        used_ids = sorted(
            {
                str(row.get("gpu_id") or "")
                for row in violations
                if isinstance(row, dict) and str(row.get("gpu_id") or "").strip()
            }
        )
        feedback = _build_gpu_boundary_feedback(
            actual_gpu_ids=used_ids,
            allowed_gpu_ids=self.gpu_ids,
            command_scope="command",
        )
        self.guard_reason[0] = reason
        self.guard_feedback[0] = feedback
        _resource_call(
            self.tool.resource_observer,
            "resource_guard_action",
            self.resource_job_id,
            action="stop_boundary_violation",
            reason=reason,
            elapsed_sec=elapsed_now,
            placement=placement,
        )
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            self.on_output(
                f"[resource guard] stopping boundary violation after "
                f"{elapsed_now:.0f}s: {reason}\n"
            )
            self.on_output(feedback.rstrip() + "\n")
        await terminate_tree(self.proc)
        await self._run_workspace_gpu_cleanup(
            reason="workspace_cleanup_after_boundary_violation",
            force=True,
        )
        return True

    def _guard_decision(
        self,
        elapsed_now: float,
        stdout_age: float,
        interval: float,
    ) -> tuple[dict | None, float]:
        parent_state = (
            "stalled_but_alive"
            if stdout_age >= max(60.0, float(self.tool.bash_heartbeat_idle_sec))
            else "healthy_running"
        )
        _resource_call(
            self.tool.resource_observer,
            "maybe_run_sidecar_backfill",
            self.resource_job_id,
            elapsed_sec=elapsed_now,
            parent_state=parent_state,
        )
        process_tree_cpu = _process_tree_cpu_snapshot(getattr(self.proc, "pid", None))
        monitor_result = _resource_call(
            self.tool.resource_observer,
            "monitor_heartbeat",
            self.resource_job_id,
            elapsed_sec=elapsed_now,
            stdout_age_sec=stdout_age,
            stdout_lines=int(self.stream_stats["stdout_lines"]),
            stdout_bytes=int(self.stream_stats["stdout_bytes"]),
            pid=getattr(self.proc, "pid", None),
            returncode=getattr(self.proc, "returncode", None),
            process_tree_cpu=process_tree_cpu,
            source="bash_guard",
        )
        if isinstance(monitor_result, dict) and monitor_result.get("recorded"):
            self.last_monitor_heartbeat_wall[0] = time.time()
        decision = _resource_call(
            self.tool.resource_observer,
            "stalled_guard_decision",
            self.resource_job_id,
            inferred_class=self.effective_resource_class,
            gpu_ids=self.gpu_ids,
            elapsed_sec=elapsed_now,
            stdout_age_sec=stdout_age,
            process_tree_cpu=process_tree_cpu,
        )
        if not isinstance(decision, dict) or not decision.get("enabled"):
            decision = {}
        interval = float(decision.get("check_interval_sec") or interval or 1.0)
        if decision.get("terminate") or decision.get("would_terminate"):
            return decision, interval
        decision = _resource_call(
            self.tool.resource_observer,
            "active_intervention_decision",
            self.resource_job_id,
            inferred_class=self.effective_resource_class,
            gpu_ids=self.gpu_ids,
            elapsed_sec=elapsed_now,
            stdout_age_sec=stdout_age,
            stdout_lines=int(self.stream_stats["stdout_lines"]),
            stdout_bytes=int(self.stream_stats["stdout_bytes"]),
            metric_history_text=str(self.stream_stats.get("metric_history_text") or ""),
            metric_history_line_count=len(
                self.stream_stats.get("metric_history_lines") or []
            ),
            saw_training_progress=bool(self.stream_stats["saw_training_progress"]),
            saw_final_score=bool(self.stream_stats["saw_final_score"]),
            current_phase=str(self.stream_stats.get("current_phase") or ""),
            invalid_metric_events=int(
                self.stream_stats.get("invalid_metric_events") or 0
            ),
            zero_score_events=int(self.stream_stats.get("zero_score_events") or 0),
            last_invalid_metric_text=str(
                self.stream_stats.get("last_invalid_metric_text") or ""
            ),
            last_zero_score_text=str(
                self.stream_stats.get("last_zero_score_text") or ""
            ),
            terminal_signal_events=int(
                self.stream_stats.get("terminal_signal_events") or 0
            ),
            terminal_signal_kind=str(
                self.stream_stats.get("terminal_signal_kind") or ""
            ),
            last_terminal_signal_text=str(
                self.stream_stats.get("last_terminal_signal_text") or ""
            ),
            process_tree_cpu=process_tree_cpu,
            **self.tool._deadline_state(),
        )
        if not isinstance(decision, dict) or not decision.get("enabled"):
            return None, interval
        return decision, float(decision.get("check_interval_sec") or interval or 1.0)

    async def _apply_guard_decision(
        self,
        decision: dict,
        elapsed_now: float,
        stdout_age: float,
    ) -> bool:
        wants_review = (
            decision.get("would_terminate")
            or decision.get("arbiter_review")
            or decision.get("requires_llm_decision")
        ) and not decision.get("terminate")
        if wants_review:
            return await self._handle_resource_review(
                decision,
                elapsed_now,
                stdout_age,
            )
        if decision.get("release_idle_lease") and not decision.get("terminate"):
            self._release_idle_guard_lease(decision, elapsed_now)
            return False
        if not decision.get("terminate"):
            return False
        reason = str(decision.get("reason") or "stalled_heavy")
        feedback = str(decision.get("feedback") or "").strip()
        self.guard_reason[0] = reason
        if feedback:
            self.guard_feedback[0] = feedback
        _resource_call(
            self.tool.resource_observer,
            "resource_guard_action",
            self.resource_job_id,
            action="terminate",
            reason=reason,
            elapsed_sec=elapsed_now,
        )
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            self.on_output(
                f"[resource guard] terminating command after "
                f"{elapsed_now:.0f}s: {reason}\n"
            )
            if feedback:
                self.on_output(feedback.rstrip() + "\n")
        await self._terminate_resource_guard_process()
        return True

    def _release_idle_guard_lease(self, decision: dict, elapsed_now: float) -> None:
        reason = str(
            decision.get("reason") or "active_intervention:idle_gpu_lease_release"
        )
        feedback = str(decision.get("feedback") or "").strip()
        release = _resource_call(
            self.tool.resource_observer,
            "release_idle_gpu_lease",
            self.resource_job_id,
            elapsed_sec=elapsed_now,
            reason=reason,
            feedback=feedback,
        )
        if self.on_output is not None:
            if self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
            state = (
                "released idle GPU lease"
                if isinstance(release, dict) and release.get("released")
                else "could not release idle GPU lease"
            )
            self.on_output(
                f"[resource guard] {state} after {elapsed_now:.0f}s: {reason}\n"
            )
            if feedback:
                self.on_output(feedback.rstrip() + "\n")
        self.guard_observe_more_until[0] = time.time() + 300.0

    async def _resource_guard(self) -> None:
        try:
            interval = 0.25
            while True:
                await asyncio.sleep(max(0.05, interval))
                if getattr(self.proc, "returncode", None) is not None:
                    return
                elapsed_now = time.time() - self.start
                stdout_age = time.monotonic() - self.io_touch[0]
                self._maybe_emit_artifact_progress(elapsed_now)
                self._emit_observe_first_health(elapsed_now, stdout_age)
                if await self._stop_gpu_boundary_violation(elapsed_now):
                    return
                decision, interval = self._guard_decision(
                    elapsed_now,
                    stdout_age,
                    interval,
                )
                if decision is not None and await self._apply_guard_decision(
                    decision,
                    elapsed_now,
                    stdout_age,
                ):
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("[bash-resource] stalled guard failed", exc_info=True)

    async def _resource_guard_watchdog(self) -> None:
        poll_sec = max(1.0, min(30.0, float(self.tool.bash_heartbeat_interval_sec or 5.0)))
        try:
            observer_interval = float(getattr(self.tool.resource_observer, "review_heartbeat_sec", 60.0) or 60.0)
        except (TypeError, ValueError):
            observer_interval = 60.0
        stale_after_sec = max(2.0 * poll_sec, 2.0 * observer_interval)
        last_gap_emit_wall: list[float] = [0.0]

        async def _record_gap_and_review(gap_reason: str, gap_sec: float) -> bool:
            elapsed_now = time.time() - self.start
            stdout_age = time.monotonic() - self.io_touch[0]
            process_tree_cpu = _process_tree_cpu_snapshot(getattr(self.proc, "pid", None))
            _resource_call(
                self.tool.resource_observer,
                "resource_monitor_gap",
                self.resource_job_id,
                elapsed_sec=elapsed_now,
                gap_sec=gap_sec,
                stdout_age_sec=stdout_age,
                stdout_lines=int(self.stream_stats["stdout_lines"]),
                stdout_bytes=int(self.stream_stats["stdout_bytes"]),
                reason=gap_reason,
                source="bash_guard_watchdog",
                process_tree_cpu=process_tree_cpu,
            )
            decision = _resource_call(
                self.tool.resource_observer,
                "active_intervention_decision",
                self.resource_job_id,
                inferred_class=self.effective_resource_class,
                gpu_ids=self.gpu_ids,
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
                process_tree_cpu=process_tree_cpu,
                **self.tool._deadline_state(),
            )
            if not isinstance(decision, dict):
                decision = {}
            if decision.get("terminate"):
                reason = str(decision.get("reason") or "resource_monitor_gap")
                feedback = str(decision.get("feedback") or "").strip()
                self.guard_reason[0] = reason
                if feedback:
                    self.guard_feedback[0] = feedback
                _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="terminate_by_monitor_watchdog", reason=reason, elapsed_sec=elapsed_now)
                if self.on_output is not None:
                    if self.heartbeat_active[0]:
                        self.on_output("\r\x1b[K")
                        self.heartbeat_active[0] = False
                    self.on_output(f"[resource watchdog] terminating command after {elapsed_now:.0f}s: {reason}\n")
                    if feedback:
                        self.on_output(feedback.rstrip() + "\n")
                await self._terminate_resource_guard_process()
                return True
            if decision.get("would_terminate") or decision.get("arbiter_review") or decision.get("requires_llm_decision"):
                _resource_call(self.tool.resource_observer, "resource_guard_action", self.resource_job_id, action="monitor_gap_review_requested", reason=str(decision.get("reason") or gap_reason), elapsed_sec=elapsed_now)
            return False

        while True:
            await asyncio.sleep(poll_sec)
            if getattr(self.proc, "returncode", None) is not None:
                return
            now_wall = time.time()
            last_wall = self.last_monitor_heartbeat_wall[0] or self.start
            gap_sec = max(0.0, now_wall - last_wall)
            task = self.guard_task
            if task is None or not task.done():
                if gap_sec >= stale_after_sec and now_wall - last_gap_emit_wall[0] >= stale_after_sec:
                    last_gap_emit_wall[0] = now_wall
                    if await _record_gap_and_review("resource_monitor_heartbeat_stale", gap_sec):
                        return
                continue
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                return
            exc_name = type(exc).__name__ if exc is not None else "completed_without_process_exit"
            self.guard_watchdog_restarts[0] += 1
            gap_reason = f"resource_guard_stopped:{exc_name}"
            if await _record_gap_and_review(gap_reason, gap_sec):
                return
            self.guard_task = asyncio.create_task(self._resource_guard())




__all__ = ["ResourceGuardMonitor"]
