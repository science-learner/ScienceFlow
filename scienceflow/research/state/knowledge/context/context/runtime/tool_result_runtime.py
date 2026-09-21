"""Stateless tool-result projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from inquirycraft.tools import ToolResult

from scienceflow.research.state.knowledge.context.context.results.base import (
    _READ_OVERLAP_COACHING,
    logger,
)
from scienceflow.research.state.knowledge.context.context.projection.bash_projection import (
    _BASH_READ_PY_SOURCE_COACHING,
    _bash_success_tail_lines_for_command,
    bash_command_dumps_python_source,
)
from scienceflow.research.state.knowledge.context.context.compression.compaction import compress_bash_tool_output_for_memory
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _parse_write_success_metadata_from_output,
    compress_edit_success_output_for_memory,
    write_success_feedback_for_memory,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import (
    _build_read_code_map_summary_for_memory,
    _build_snapshot_ref_output_for_memory,
    _read_args_has_explicit_range,
    _read_args_limit,
    compress_read_output_for_memory,
)
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _build_redundant_read_coverage_summary,
    _build_write_auto_snapshot_block,
    _merge_read_intervals,
    _numbered_source_ranges_in_text,
    _requested_read_range_for_file,
    _sha256_short_bytes,
)


@dataclass
class ToolResultProjection:
    rel: str
    observation: str
    command: str
    bare_solution_bash: bool
    signature: tuple[str, str] | None
    repeated: bool
    read_overlap_redundant: bool
    read_overlap_summary: str | None


def _update_tool_result_state(
    manager: Any,
    tool_name: str,
    args: dict[str, Any],
    result: ToolResult,
    *,
    rel: str,
    raw_id: str,
) -> tuple[bool, str | None]:
    if tool_name == "edit" and result.error:
        manager._edit_fail_counter[rel] = manager._edit_fail_counter.get(rel, 0) + 1
        manager._edit_fail_total_counter[rel] = manager._edit_fail_total_counter.get(rel, 0) + 1
    elif tool_name == "edit":
        manager._edit_fail_counter.pop(rel, None)
        manager._edit_fail_total_counter.pop(rel, None)
        manager._edit_fail_escalation_level.pop(rel, None)
    if tool_name in ("write", "edit") and not result.error and rel:
        manager._read_coverage.pop(rel, None)
        try:
            resolved = manager._guard.resolve(rel)
            if resolved.is_file():
                manager._prune_symbol_read_coverage_after_write(
                    rel,
                    resolved.read_text(encoding="utf-8", errors="replace"),
                )
        except (ValueError, OSError):
            manager._read_symbol_coverage.pop(rel, None)
        manager._update_snapshot_from_path(rel)
    if (
        tool_name == "read"
        and not result.error
        and rel
        and _read_args_has_explicit_range(args)
        and _read_args_limit(args) > 0
    ):
        return manager._update_read_coverage_and_is_redundant(
            rel,
            args,
            raw_id=raw_id,
        )
    return False, None


def _append_edit_failure_feedback(
    manager: Any,
    observation: str,
    *,
    rel: str,
    edit_failed: bool,
) -> str:
    if not edit_failed:
        return observation
    if manager._edit_fail_counter.get(rel, 0) >= 2:
        try:
            observation += "\n\n" + manager.inject_file_content(
                rel, reason="edit failed twice"
            )
            manager._edit_fail_counter[rel] = 0
        except OSError as error:
            logger.debug("auto-read after edit fail: %s", error)
    total_fails = manager._edit_fail_total_counter.get(rel, 0)
    previous_level = manager._edit_fail_escalation_level.get(rel, 0)
    level = 0
    coaching = ""
    if total_fails >= 6:
        level = 2
        coaching = (
            "[Guard] edit has failed repeatedly on this file (6+ times). "
            "Stop retrying `edit` with guessed snippets. You MUST `read` the latest file "
            "content and then switch to a full `write` (entire file body) as fallback."
        )
    elif total_fails >= 4:
        level = 1
        coaching = (
            "[Guard] edit has failed multiple times on this file (4+ times). "
            "Re-sync context before the next attempt: `read` the file and either use a "
            "larger exact `old_str` span or fallback to `write` full-file replacement."
        )
    if level > previous_level and coaching:
        observation += "\n\n" + coaching
        manager._edit_fail_escalation_level[rel] = level
    return observation


def _compress_read_observation(
    manager: Any,
    projection: ToolResultProjection,
    args: dict[str, Any],
    *,
    raw_id: str,
) -> None:
    if projection.repeated and projection.signature is not None:
        manager._released_full_signatures.add(projection.signature)
        return
    if _read_args_has_explicit_range(args) and _read_args_limit(args) > 0:
        return
    replaced = False
    if projection.rel and manager._path_matches_write_snapshot(projection.rel):
        try:
            resolved = manager._guard.resolve(projection.rel)
            if resolved.is_file():
                text = resolved.read_text(encoding="utf-8", errors="replace")
                sha = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
                if sha and sha == manager._last_auto_snapshot_sha_by_path.get(projection.rel, ""):
                    projection.observation = _build_snapshot_ref_output_for_memory(
                        projection.observation,
                        rel=projection.rel,
                        sha=sha,
                        raw_id=raw_id,
                    )
                    replaced = True
                elif (
                    PurePosixPath(projection.rel).suffix.lower()
                    in manager._write_auto_snapshot_code_extensions
                    and len(text.splitlines()) > manager._read_success_max_lines
                ):
                    projection.observation = _build_read_code_map_summary_for_memory(
                        projection.observation,
                        rel=projection.rel,
                        text=text,
                        raw_id=raw_id,
                    )
                    replaced = True
        except (ValueError, OSError) as error:
            logger.debug("read code-map/snapshot compression skipped: %s", error)
    if not replaced:
        projection.observation = compress_read_output_for_memory(
            projection.observation,
            max_lines=manager._read_success_max_lines,
        )


def _compress_tool_observation(
    manager: Any,
    projection: ToolResultProjection,
    tool_name: str,
    args: dict[str, Any],
    result: ToolResult,
    *,
    raw_id: str,
) -> None:
    if not manager._tool_memory_compression:
        return
    if tool_name == "edit" and not result.error:
        projection.observation = compress_edit_success_output_for_memory(
            projection.observation
        )
    elif tool_name == "write" and not result.error:
        lines, sha = _parse_write_success_metadata_from_output(projection.observation)
        projection.observation = write_success_feedback_for_memory(
            projection.rel,
            lines=lines,
            sha256_short=sha,
        )
    elif tool_name == "bash" and not result.error:
        if projection.repeated and projection.signature is not None and not projection.bare_solution_bash:
            manager._released_full_signatures.add(projection.signature)
            return
        tail_lines = (
            min(manager._bash_success_tail_lines, manager._bash_success_tail_lines_solution)
            if projection.bare_solution_bash
            else _bash_success_tail_lines_for_command(
                projection.command,
                fallback=manager._bash_success_tail_lines,
                solution=manager._bash_success_tail_lines_solution,
                test=manager._bash_success_tail_lines_test,
                readonly=manager._bash_success_tail_lines_readonly,
                install=manager._bash_success_tail_lines_install,
            )
        )
        projection.observation = compress_bash_tool_output_for_memory(
            projection.observation,
            tail_lines=tail_lines,
            command=projection.command,
            dedup_enabled=manager._bash_output_dedup_apply_to_memory,
            dedup_min_repeat=manager._bash_output_dedup_min_repeat,
            dedup_summary_prefix=manager._bash_output_dedup_summary_prefix,
        )
        if not projection.bare_solution_bash and bash_command_dumps_python_source(projection.command):
            projection.observation += _BASH_READ_PY_SOURCE_COACHING
    elif tool_name == "read" and not result.error:
        _compress_read_observation(manager, projection, args, raw_id=raw_id)
    elif tool_name == "grep" and not result.error:
        projection.observation = manager._compress_grep_snapshot_refs_for_memory(
            projection.observation,
            raw_id=raw_id,
        )


def _apply_redundant_read_projection(
    manager: Any,
    projection: ToolResultProjection,
    args: dict[str, Any],
) -> None:
    if not projection.read_overlap_redundant:
        return
    rel = projection.rel
    manager._redundant_read_count_by_path[rel] = (
        manager._redundant_read_count_by_path.get(rel, 0) + 1
    )
    replaced = False
    if projection.read_overlap_summary:
        first_newline = projection.observation.find("\n")
        header = (
            projection.observation
            if first_newline < 0
            else projection.observation[:first_newline]
        )
        projection.observation = (
            header.rstrip() + "\n\n" + projection.read_overlap_summary
        ).strip()
        replaced = True
    if not replaced and manager._write_auto_snapshot_enabled and rel and manager._path_matches_write_snapshot(rel):
        try:
            resolved = manager._guard.resolve(rel)
            if resolved.is_file():
                text = resolved.read_text(encoding="utf-8", errors="replace")
                sha = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
                covered = list((manager._read_coverage.get(rel) or ("", []))[1])
                requested = _requested_read_range_for_file(args, len(text.splitlines()))
                first_newline = projection.observation.find("\n")
                header = (
                    projection.observation
                    if first_newline < 0
                    else projection.observation[:first_newline]
                )
                projection.observation = (
                    header.rstrip()
                    + "\n\n"
                    + _build_redundant_read_coverage_summary(
                        rel,
                        sha=sha,
                        requested=requested,
                        covered_ranges=covered,
                    )
                ).strip()
                replaced = True
        except (ValueError, OSError) as error:
            logger.debug(
                "silent read intercept: coverage summary failed (%s)",
                error,
                exc_info=True,
            )
    if not replaced:
        projection.observation += _READ_OVERLAP_COACHING


def _append_write_snapshot(
    manager: Any,
    projection: ToolResultProjection,
    tool_name: str,
    args: dict[str, Any],
    result: ToolResult,
) -> None:
    rel = projection.rel
    if not (
        manager._tool_memory_compression
        and tool_name in ("write", "edit")
        and not result.error
        and rel
        and manager._write_auto_snapshot_enabled
        and manager._path_matches_write_snapshot(rel)
    ):
        return
    try:
        resolved = manager._guard.resolve(rel)
        if not resolved.is_file():
            return
        try:
            current_text = resolved.read_text(encoding="utf-8", errors="replace")
            current_sha = _sha256_short_bytes(current_text.encode("utf-8", errors="replace"))
        except OSError:
            current_text = current_sha = ""
        previous_sha = manager._last_auto_snapshot_sha_by_path.get(rel, "")
        if current_sha and current_sha == previous_sha:
            logger.debug(
                "%s auto-snapshot: skipped (sha %s unchanged for %s)",
                tool_name,
                current_sha,
                rel,
            )
            return
        snapshot = _build_write_auto_snapshot_block(
            rel,
            resolved,
            max_lines=manager._write_auto_snapshot_max_lines,
            max_chars=manager._write_auto_snapshot_max_chars,
            previous_text=manager._last_auto_snapshot_text_by_path.get(rel),
            tool_name=tool_name,
            args=args,
            changed_context_lines=manager._write_auto_snapshot_changed_context_lines,
            symbol_body_lines=manager._write_auto_snapshot_symbol_body_lines,
        )
        projection.observation = projection.observation.rstrip() + "\n\n" + snapshot
        if current_sha:
            manager._last_auto_snapshot_sha_by_path[rel] = current_sha
            manager._last_auto_snapshot_text_by_path[rel] = current_text
            ranges = _numbered_source_ranges_in_text(snapshot)
            if ranges:
                manager._read_coverage[rel] = (current_sha, _merge_read_intervals(ranges))
    except (ValueError, OSError) as error:
        logger.debug("%s auto-snapshot: skipped (%s)", tool_name, error, exc_info=True)
