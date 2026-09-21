# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: managed resource waiting and wake feedback."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    asyncio,
    time,
)


class WaitRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def resource_wait_instruction_for_bash_sleep(
        self,
        *,
        command: str = "",
        planned_sleep_sec: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        records: list[dict[str, Any]] = []
        for record in self._resource_wait_tokens.values():
            if not isinstance(record, dict):
                continue
            try:
                created_at = float(record.get("created_at") or 0.0)
                max_wait = float(record.get("max_wait_sec") or 0.0)
            except (TypeError, ValueError):
                continue
            if created_at <= 0:
                continue
            # Tokens are intended for the immediate RESOURCE_FEEDBACK turn. If
            # an old token survives past its wait window, do not hijack unrelated
            # sleep commands.
            if max_wait > 0 and now - created_at > max_wait:
                continue
            records.append(record)
        if not records:
            return {"available": False, "reason": "no_active_resource_wait_option"}

        record = max(records, key=lambda r: float(r.get("created_at") or 0.0))
        wait_max = max(0.05, min(float(record.get("max_wait_sec") or 600.0), 1800.0))
        if planned_sleep_sec is not None:
            try:
                wait_max = max(0.05, min(wait_max, float(planned_sleep_sec)))
            except (TypeError, ValueError):
                pass
        payload = {
            **record,
            "command": str(command or ""),
            "planned_sleep_sec": planned_sleep_sec,
            "recommended_tool": "resource_wait",
        }
        self.record_resource_event(
            "managed_resource_wait_bash_sleep_blocked",
            payload=payload,
            command_id=str(record.get("job_id") or ""),
            lease_id=str(record.get("job_id") or ""),
        )
        feedback = self._resource_wait_feedback(
            status="RESOURCE_WAIT_REQUIRED",
            reason="use resource_wait tool instead of bash sleep",
            record=record,
            extra_facts={
                "retry_allowed": "false",
                "same_command_retry_allowed": "false",
                "action_options": "resource_wait,cpu_support",
                "cpu_support_preferred": "true",
                "wait_is_optional": "true",
                "bash_sleep_allowed": "false",
                "wait_tool": "resource_wait",
                "wait_token": str(record.get("wait_token") or ""),
                "wait_max_sec": "%.0f" % wait_max,
                "wait_reason": str(record.get("reason") or "resource_busy"),
            },
        )
        return {
            "available": True,
            "wait_token": str(record.get("wait_token") or ""),
            "wait_max_sec": wait_max,
            "feedback": feedback,
        }


    async def managed_resource_wait(
        self,
        *,
        wait_token: str,
        max_wait_sec: float | None = None,
        reason: str = "resource_busy",
    ) -> dict[str, Any]:
        token = str(wait_token or "").strip()
        record = self._resource_wait_tokens.get(token)
        if not isinstance(record, dict):
            feedback = self._resource_wait_feedback(
                status="RESOURCE_WAIT_NOT_AVAILABLE",
                reason="no active resource wait option",
                extra_facts={"retry_allowed": "false"},
            )
            self.record_resource_event("managed_resource_wait_invalid", payload={"wait_token": token, "reason": "invalid_or_expired"})
            return {"status": "RESOURCE_WAIT_NOT_AVAILABLE", "reason": "invalid_or_expired", "feedback": feedback}
        wait_max = max(0.05, min(float(max_wait_sec or record.get("max_wait_sec") or 600.0), float(record.get("max_wait_sec") or 600.0), 1800.0))
        started_at = time.monotonic()
        self.record_resource_event(
            "managed_resource_wait_started",
            payload={**record, "requested_max_wait_sec": wait_max, "reason": str(reason or record.get("reason") or "resource_busy")},
            command_id=str(record.get("job_id") or ""),
            lease_id=str(record.get("job_id") or ""),
        )
        while True:
            wake_reason = self._resource_wait_wake_reason(record)
            if wake_reason:
                self._resource_wait_tokens.pop(token, None)
                elapsed = max(0.0, time.monotonic() - started_at)
                payload = {
                    **record,
                    "elapsed_sec": elapsed,
                    "wake_reason": wake_reason,
                    "wake_pressure_generation": self._current_pressure_generation(),
                }
                self.record_resource_event(
                    "managed_resource_wait_woken",
                    payload=payload,
                    command_id=str(record.get("job_id") or ""),
                    lease_id=str(record.get("job_id") or ""),
                )
                feedback = self._resource_wait_feedback(
                    status="RESOURCE_AVAILABLE",
                    reason=wake_reason,
                    record=record,
                    extra_facts={
                        "retry_allowed": "true",
                        "same_command_retry_allowed": "true",
                        "action_options": "retry_after_feedback_change",
                        "wake_reason": wake_reason,
                        "delta_since_wait_started": wake_reason,
                    },
                )
                return {"status": "RESOURCE_AVAILABLE", "reason": wake_reason, "elapsed_sec": elapsed, "feedback": feedback}
            elapsed = time.monotonic() - started_at
            if elapsed >= wait_max:
                self._resource_wait_tokens.pop(token, None)
                scope_key = str(record.get("scope_key") or "")
                if scope_key:
                    suppress_sec = max(60.0, min(float(record.get("max_wait_sec") or wait_max or 600.0), 600.0))
                    self._resource_wait_suppressed_until[scope_key] = time.time() + suppress_sec
                payload = {**record, "elapsed_sec": elapsed, "timeout_fuse": True}
                self.record_resource_event(
                    "managed_resource_wait_timeout",
                    payload=payload,
                    command_id=str(record.get("job_id") or ""),
                    lease_id=str(record.get("job_id") or ""),
                )
                feedback = self._resource_wait_feedback(
                    status="RESOURCE_WAIT_TIMEOUT",
                    reason="no material resource state change",
                    record=record,
                    extra_facts={
                        "retry_allowed": "false",
                        "same_command_retry_allowed": "false",
                        "resource_wait_allowed": "false",
                        "action_options": "cpu_support,replan,retry_after_feedback_change",
                    },
                )
                return {"status": "RESOURCE_WAIT_TIMEOUT", "reason": "timeout", "elapsed_sec": elapsed, "feedback": feedback}
            await asyncio.sleep(min(0.1, max(0.05, wait_max - elapsed)))
