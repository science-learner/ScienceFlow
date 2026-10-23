"""Streaming output and progress projection for a running Bash session."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from scienceflow.runtime.core.process import ProcessStream, ProcessStreamChunk
from scienceflow.runtime.safety.resource.review.signals import scan_output_health
from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _FINAL_VALIDATION_SCORE_RE,
    _TRAINING_PROGRESS_RE,
    _VALIDATION_OR_INFERENCE_RE,
    _resource_call,
)
from scienceflow.runtime.safety.tooling.workspace.shell_output import (
    _sanitize_model_visible_output_paths,
)
from scienceflow.runtime.safety.tooling.resource_management.signals import (
    _latest_artifact_snapshot,
    _parse_progress_signals,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.metric_history import (
    metric_history_text,
    update_metric_history_lines,
)


class BashStreamMonitor:
    """Own live-output caps, heartbeats, and progress/artifact projections."""

    def __init__(
        self,
        tool: Any,
        *,
        workspace_dir: Any,
        resource_job_id: str | None,
        run_artifact_dir: Any,
        run_state_path: Any,
        started_at: float,
        started_at_wall: float,
        on_output: Any,
    ) -> None:
        self.tool = tool
        self.workspace_dir = workspace_dir
        self.resource_job_id = resource_job_id
        self.run_artifact_dir = run_artifact_dir
        self.run_state_path = run_state_path
        self.started_at = started_at
        self.started_at_wall = started_at_wall
        self.on_output = on_output
        self.stream_state: dict[str, int | bool] = {"sent": 0, "truncated": False}
        self.line_cap = int(tool.max_stream_line_chars)
        self.total_cap = int(tool.max_output_chars)
        self.idle_after = float(tool.bash_heartbeat_idle_sec)
        self.heartbeat_interval = float(tool.bash_heartbeat_interval_sec)
        self.io_touch = [time.monotonic()]
        self.last_heartbeat_emit = [0.0]
        self.heartbeat_active = [False]
        self.stats: dict[str, Any] = {
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "saw_training_progress": False,
            "saw_final_score": False,
            "current_phase": "",
            "invalid_metric_events": 0,
            "zero_score_events": 0,
            "last_invalid_metric_text": "",
            "last_zero_score_text": "",
            "terminal_signal_events": 0,
            "terminal_signal_kind": "",
            "last_terminal_signal_text": "",
            "last_progress_signals": {},
            "last_progress_emitted": {},
            "metric_history_lines": [],
            "metric_history_text": "",
            "last_artifact": {},
            "last_artifact_emitted": {},
            "last_artifact_emitted_key": (),
        }
        self.last_progress_emit = [0.0]
        self.last_artifact_scan = [0.0]
        self.artifact_watch_cache: dict[str, Any] = {}
        self.progress_min_interval = max(
            0.0,
            float(tool.resource_progress_heartbeat_min_interval_sec or 0.0),
        )
        self.artifact_scan_interval = max(
            0.05,
            float(tool.resource_artifact_heartbeat_scan_interval_sec or 30.0),
        )

    def touch_io(self) -> None:
        self.io_touch[0] = time.monotonic()
        self.last_heartbeat_emit[0] = 0.0

    def emit_progress(
        self,
        progress_signals: dict[str, Any],
        *,
        elapsed_sec: float | None = None,
        force: bool = False,
        reason: str = "progress",
    ) -> None:
        if not progress_signals:
            return
        now = time.time()
        changed = progress_signals != (self.stats.get("last_progress_emitted") or {})
        if self.last_progress_emit[0] <= 0.0:
            emit_reason = "initial"
        elif force and changed:
            emit_reason = reason or "final"
        elif changed and (
            self.progress_min_interval <= 0.0
            or now - self.last_progress_emit[0] >= self.progress_min_interval
        ):
            emit_reason = "interval"
        else:
            return
        self.last_progress_emit[0] = now
        self.stats["last_progress_emitted"] = dict(progress_signals)
        _resource_call(
            self.tool.resource_observer,
            "progress_heartbeat",
            self.resource_job_id,
            elapsed_sec=max(
                0.0,
                float(elapsed_sec)
                if elapsed_sec is not None
                else now - self.started_at,
            ),
            phase=str(self.stats.get("current_phase") or "training"),
            signals=progress_signals,
            stdout_lines=int(self.stats["stdout_lines"]),
            stdout_bytes=int(self.stats["stdout_bytes"]),
            metric_history_text=str(self.stats.get("metric_history_text") or ""),
            metric_history_line_count=len(self.stats.get("metric_history_lines") or []),
            emit_reason=emit_reason,
        )

    def update_stats(self, text: str, *, stderr: bool = False) -> None:
        if not text:
            return
        if not stderr:
            self.stats["stdout_bytes"] = int(self.stats["stdout_bytes"]) + len(
                text.encode(errors="replace")
            )
            self.stats["stdout_lines"] = int(self.stats["stdout_lines"]) + len(
                text.splitlines() or [text]
            )
        if _FINAL_VALIDATION_SCORE_RE.search(text):
            self.stats["saw_final_score"] = True
            self.stats["current_phase"] = "final_scoring"
        if _TRAINING_PROGRESS_RE.search(text):
            self.stats["saw_training_progress"] = True
            if not self.stats.get("current_phase"):
                self.stats["current_phase"] = "training"
        if _VALIDATION_OR_INFERENCE_RE.search(text):
            self.stats["current_phase"] = "validation_or_inference"
        health = scan_output_health(text)
        lines = update_metric_history_lines(
            self.stats.get("metric_history_lines") or [],
            text,
        )
        if lines != self.stats.get("metric_history_lines"):
            self.stats["metric_history_lines"] = lines
            self.stats["metric_history_text"] = metric_history_text(lines)
        for key, text_key in (
            ("invalid_metric_events", "last_invalid_metric_text"),
            ("zero_score_events", "last_zero_score_text"),
            ("terminal_signal_events", "last_terminal_signal_text"),
        ):
            if health.get(key):
                self.stats[key] = int(self.stats.get(key) or 0) + int(
                    health.get(key) or 0
                )
                self.stats[text_key] = str(health.get(text_key) or "")
        if health.get("invalid_metric_events") or health.get("zero_score_events"):
            self.stats["saw_training_progress"] = True
        if health.get("terminal_signal_kind"):
            self.stats["terminal_signal_kind"] = str(
                health.get("terminal_signal_kind") or ""
            )
        progress = _parse_progress_signals(text)
        if progress:
            self.stats["saw_training_progress"] = True
            self.stats["last_progress_signals"] = progress
            phase = str(progress.get("phase") or "").strip()
            if phase:
                self.stats["current_phase"] = phase
            elif not self.stats.get("current_phase"):
                self.stats["current_phase"] = "training"
            self.emit_progress(
                progress,
                reason="scienceflow_hb" if "heartbeat" in progress else "progress",
            )

    def emit_artifact(self, elapsed_sec: float, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_artifact_scan[0] < self.artifact_scan_interval:
            return
        self.last_artifact_scan[0] = now
        artifact = _latest_artifact_snapshot(
            self.workspace_dir,
            started_at=self.started_at_wall,
            run_artifact_dir=self.run_artifact_dir,
            run_state_path=self.run_state_path,
            stability_cache=self.artifact_watch_cache,
            settle_sec=float(self.tool.resource_artifact_recoverable_settle_sec or 0.0),
        )
        if not artifact:
            return
        key = (
            str(artifact.get("path") or ""),
            float(artifact.get("mtime") or 0.0),
            int(artifact.get("size_bytes") or 0),
            str(artifact.get("stability") or ""),
            bool(artifact.get("recoverable_artifact_on_disk")),
        )
        self.stats["last_artifact"] = artifact
        if key == self.stats.get("last_artifact_emitted_key"):
            return
        self.stats["last_artifact_emitted"] = artifact
        self.stats["last_artifact_emitted_key"] = key
        signals = dict(self.stats.get("last_progress_signals") or {})
        signals["artifact"] = artifact
        self.stats["last_progress_signals"] = signals
        _resource_call(
            self.tool.resource_observer,
            "progress_heartbeat",
            self.resource_job_id,
            elapsed_sec=max(0.0, float(elapsed_sec or 0.0)),
            phase=str(self.stats.get("current_phase") or "artifact_update"),
            signals=signals,
            stdout_lines=int(self.stats["stdout_lines"]),
            stdout_bytes=int(self.stats["stdout_bytes"]),
            emit_reason="artifact",
        )

    async def emit_line(self, text: str, prefix: str) -> None:
        if self.on_output is None or self.stream_state["truncated"]:
            return
        if self.heartbeat_active[0]:
            self.on_output("\r\033[K")
            self.heartbeat_active[0] = False
        shown_text = _sanitize_model_visible_output_paths(
            text,
            self.workspace_dir,
            self.tool.path_guard_extra_roots or (),
            strip_symlink_targets=bool(self.tool.strip_symlink_targets),
        )
        if self.line_cap > 0 and len(shown_text) > self.line_cap:
            shown_text = shown_text[: self.line_cap] + "… [line truncated]\n"
        chunk = f"{prefix}{shown_text}"
        if self.total_cap > 0 and int(self.stream_state["sent"]) + len(chunk) > self.total_cap:
            room = self.total_cap - int(self.stream_state["sent"])
            if room > 120:
                self.on_output(chunk[:room])
            displayed = int(self.stream_state["sent"]) + min(len(chunk), max(0, room))
            self.on_output(
                f"{prefix}… [live stdout truncated: displayed ~{displayed} chars "
                "of stream mirror; full capture still used for tool result "
                f"(trimmed to max_output_chars={self.total_cap})]\n"
            )
            self.stream_state["truncated"] = True
            return
        self.stream_state["sent"] = int(self.stream_state["sent"]) + len(chunk)
        self.on_output(chunk)

    async def consume_chunk(self, chunk: ProcessStreamChunk) -> None:
        prefix = "[stderr] " if chunk.stream is ProcessStream.STDERR else ""
        self.touch_io()
        self.update_stats(chunk.text, stderr=bool(prefix))
        _resource_call(
            self.tool.resource_observer,
            "stdout_heartbeat",
            self.resource_job_id,
        )
        await self.emit_line(chunk.text, prefix)

    async def heartbeat(self, timeout: float) -> None:
        while True:
            await asyncio.sleep(0.25)
            now = time.monotonic()
            elapsed = time.time() - self.started_at_wall
            if now - self.io_touch[0] < self.idle_after:
                continue
            if self.last_heartbeat_emit[0] > 0 and (
                now - self.last_heartbeat_emit[0]
            ) < self.heartbeat_interval:
                continue
            self.last_heartbeat_emit[0] = now
            self.on_output(
                f"\r\033[K[bash … {elapsed:.0f}s elapsed, "
                f"{max(0.0, timeout - elapsed):.0f}s until timeout]"
            )
            self.heartbeat_active[0] = True


__all__ = ("BashStreamMonitor",)
