"""Stable memory-context facade over pure and stateful responsibilities."""

from __future__ import annotations

import re as re

from scienceflow.research.state.knowledge.context.context.results.base import (
    COMPACTED_CONVERSATION_SUMMARY_MARKER as COMPACTED_CONVERSATION_SUMMARY_MARKER,
    _READ_OVERLAP_COACHING as _READ_OVERLAP_COACHING,
)
from scienceflow.research.state.knowledge.context.context.projection.bash_projection import (
    _bash_is_pure_read_only as _bash_is_pure_read_only,
    bash_command_dumps_python_source as bash_command_dumps_python_source,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_compression import (
    apply_clone_inherit_compression as apply_clone_inherit_compression,
    apply_clone_minimal_slice_to_memory as apply_clone_minimal_slice_to_memory,
    apply_clone_non_a_memory_budget as apply_clone_non_a_memory_budget,
    build_shallow_continuity_capsule as build_shallow_continuity_capsule,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_inheritance import (
    _compile_clone_inherit_signal_patterns as _compile_clone_inherit_signal_patterns,
    compress_inherited_tool_message_content as compress_inherited_tool_message_content,
    slice_clone_minimal_inherited_messages as slice_clone_minimal_inherited_messages,
)
from scienceflow.research.state.knowledge.context.context.compression.compaction import (
    compress_bash_tool_output_for_memory as compress_bash_tool_output_for_memory,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _best_suffix_for_budget_priority as _best_suffix_for_budget_priority,
    _ensure_complete_tool_turn_prefix as _ensure_complete_tool_turn_prefix,
    _finalize_messages_for_llm as _finalize_messages_for_llm,
    _message_char_len as _message_char_len,
    _message_priority_score as _message_priority_score,
    _sanitize_orphan_tool_messages as _sanitize_orphan_tool_messages,
    compress_edit_success_output_for_memory as compress_edit_success_output_for_memory,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import (
    compress_read_output_for_memory as compress_read_output_for_memory,
    trim_tool_feedback_for_llm_context as trim_tool_feedback_for_llm_context,
)
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX as _AUTO_SNAPSHOT_PREFIX,
    _build_write_auto_snapshot_block as _build_write_auto_snapshot_block,
)
from scienceflow.research.state.knowledge.context.context.runtime import manager_state
from scienceflow.research.state.knowledge.context.context.projection import manager_projection
from scienceflow.research.state.knowledge.context.context.compression import manager_compaction
from scienceflow.research.state.knowledge.context.context.runtime import manager_tool_results
from scienceflow.research.state.knowledge.context.context.runtime import manager_finalize


class MemoryContextManager:
    """State owner for one agent memory context."""
    __init__ = manager_state.__init__
    _norm_rel = manager_state._norm_rel
    _path_matches_write_snapshot = manager_state._path_matches_write_snapshot
    _current_snapshot_sha_for_rel = manager_state._current_snapshot_sha_for_rel
    _compress_grep_snapshot_refs_for_memory = manager_state._compress_grep_snapshot_refs_for_memory
    _update_read_coverage_and_is_redundant = manager_state._update_read_coverage_and_is_redundant
    _prune_symbol_read_coverage_after_write = manager_state._prune_symbol_read_coverage_after_write
    _trim_snapshots = manager_state._trim_snapshots
    _update_snapshot_from_path = manager_projection._update_snapshot_from_path
    file_state_summary = manager_projection.file_state_summary
    pin_message = manager_projection.pin_message
    pinned_messages = manager_projection.pinned_messages
    build_messages_for_llm_with_stats = manager_projection.build_messages_for_llm_with_stats
    build_messages_for_llm = manager_projection.build_messages_for_llm
    rewrite_messages = manager_projection.rewrite_messages
    replay_state_from_inherited_memory = manager_projection.replay_state_from_inherited_memory
    mechanical_compress_old_messages = manager_compaction.mechanical_compress_old_messages
    _leading_user_messages_for_compact = manager_compaction._leading_user_messages_for_compact
    set_protected_raw_prefix = manager_compaction.set_protected_raw_prefix
    _protected_raw_prefix_parts = manager_compaction._protected_raw_prefix_parts
    replace_protected_raw_prefix_with_summary = manager_compaction.replace_protected_raw_prefix_with_summary
    build_inband_compact_messages = manager_compaction.build_inband_compact_messages
    replace_history_with_compacted_summary = manager_compaction.replace_history_with_compacted_summary
    record_tool_result = manager_tool_results.record_tool_result
    inject_file_content = manager_finalize.inject_file_content
    compact = manager_finalize.compact
