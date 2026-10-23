"""ScienceFlow policies adapted to InquiryCraft's post-commit lifecycle."""

from __future__ import annotations

import inspect
from typing import Any

from inquirycraft.runtime import (
    RuntimeContext,
    ToolCommitDecision,
    ToolCommitObservation,
)
from inquirycraft.tools import ToolOutputReducerRegistry, ToolResult, coerce_tool_result

from scienceflow.agent.core.ports.callback_ports import resolve_agent_callback


def scienceflow_output_reducers(host: Any) -> ToolOutputReducerRegistry:
    """Make IQ the single projection owner while preserving SF-visible feedback."""

    registry = ToolOutputReducerRegistry()

    def register(tool_name: str) -> None:
        def reduce(*, args, result, **_):
            guard = getattr(host, "_guard_manager", None)
            coaching = (
                guard.process_tool_result(tool_name, args, result) if guard else ""
            )
            visible = host._host_ports.prepare_tool_feedback_for_memory(
                tool_name, args, result, guard_coaching=coaching
            )
            return visible, "scienceflow_feedback_v1"

        registry.register(tool_name, reduce)

    for tool in host.availableTools:
        register(str(tool.name))
    return registry


class ScienceFlowPostCommitAdapter:
    """Run domain callbacks from immutable, journaled tool evidence."""

    def __init__(self, host: Any) -> None:
        self.host = host
        self._seen: set[tuple[str, str, int]] = set()

    async def __call__(
        self, observation: ToolCommitObservation, context: RuntimeContext
    ) -> ToolCommitDecision | None:
        if observation.recovering and not observation.replayed:
            return None
        key = (
            context.run_id,
            str(observation.invocation.id),
            observation.journal_attempt,
        )
        if key in self._seen:
            return None
        self._seen.add(key)
        host = self.host
        name = str(observation.invocation.name)
        args = host._host_ports.normalize_tool_input_paths(
            name, dict(observation.invocation.arguments)
        )
        args.pop("thought", None)
        raw_result = coerce_tool_result(observation.raw_result)
        await self._candidate_archive(name, args, raw_result)
        if name != "bash":
            return None
        early = await host._host_ports.maybe_embedded_full_run_after_quick_test(
            args, raw_result
        )
        if early is not None:
            return self._terminate(str(early), "embedded_full_run")
        await host._host_ports.maybe_write_bare_run_tail_snapshot(args, raw_result)
        stage_callback = resolve_agent_callback(host, "stage_capture")
        if not self._stage_ready(raw_result, stage_callback):
            return None
        stage_output = stage_callback(agent=host, args=args, tool_result=raw_result)
        if inspect.isawaitable(stage_output):
            stage_output = await stage_output
        self._refresh_stage_id(stage_callback, context)
        if stage_output:
            return self._terminate(str(stage_output), "stage_capture")
        return None

    async def _candidate_archive(
        self, name: str, args: dict[str, Any], result: ToolResult
    ) -> None:
        callback = resolve_agent_callback(self.host, "candidate_archive")
        if not callable(callback):
            return
        try:
            value = callback(
                agent=self.host,
                tool_name=name,
                args=args,
                tool_result=result,
            )
            if inspect.isawaitable(value):
                await value
        except Exception as exc:
            self.host._log_info(
                "[long-horizon-flow] candidate artifact archive callback failed: %s",
                exc,
            )

    def _stage_ready(self, result: ToolResult, callback: Any) -> bool:
        host = self.host
        owner = getattr(callback, "__self__", None)
        if isinstance(getattr(owner, "pending_text_stage_commit", None), dict):
            return False
        artifact = str(getattr(host, "_lnr_candidate_artifact_rel", "") or "").strip()
        artifact_ready = bool(
            getattr(host, "_lnr_stage_capture_on_candidate_artifact", False)
            and artifact
            and (host._workspace_dir / artifact).exists()
        )
        return bool(
            callable(callback)
            and not host._embedded_full_run_enabled
            and not result.error
            and (getattr(host, "_lnr_snapshot_ok", False) or artifact_ready)
        )

    @staticmethod
    def _refresh_stage_id(callback: Any, context: RuntimeContext) -> None:
        owner = getattr(callback, "__self__", None)
        snapshots = getattr(owner, "stage_snapshots", ())
        latest = None
        fallback_stage_id = ""
        if isinstance(snapshots, dict) and snapshots:
            fallback_stage_id = str(next(reversed(snapshots)))
            latest = snapshots[fallback_stage_id]
        elif snapshots:
            latest = snapshots[-1]
        next_stage = getattr(owner, "_next_stage_id_for_logging", None)
        if callable(next_stage):
            stage_id = str(next_stage() or "draft")
        elif latest is not None:
            stage_id = str(
                getattr(latest, "stage_id", "") or fallback_stage_id or "draft"
            )
        else:
            return
        lineage = getattr(owner, "_lineage_uid_prefix", None)
        lineage_id = str(lineage() or "") if callable(lineage) else ""
        node = getattr(owner, "_stage_node_uid", None)
        node_uid = (
            str(node(stage_id, lineage_id=lineage_id) or "") if callable(node) else ""
        )
        context.metadata["stage_id"] = stage_id
        context.metadata["lineage_id"] = lineage_id
        context.metadata["node_uid"] = node_uid
        host = getattr(callback, "__self__", None)
        agent = getattr(host, "_resource_main_agent_ref", None)
        if agent is not None:
            agent._scienceflow_stage_id = stage_id
            agent._scienceflow_lineage_id = lineage_id
            agent._scienceflow_node_uid = node_uid

    @staticmethod
    def _terminate(output: str, route: str) -> ToolCommitDecision:
        return ToolCommitDecision(
            action="terminate_session",
            output=output,
            reason=f"scienceflow_{route}",
            route=route,
        )


__all__ = ["ScienceFlowPostCommitAdapter", "scienceflow_output_reducers"]
