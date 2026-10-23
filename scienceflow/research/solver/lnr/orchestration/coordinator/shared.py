# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Canonical LNR composition coordinator.

Domain state, policy, effects, and lifecycle behavior live in independently
owned modules. This coordinator retains cross-component orchestration and the
public ``LnrSolver`` surface while the deprecated import path remains a
module-object facade.
"""

from __future__ import annotations

__scienceflow_canonical_backend__ = True

import asyncio
import copy
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from inquirycraft.memory import Message
from inquirycraft.llm import StreamHandle
from inquirycraft.tools import ToolResult

from scienceflow.agent.core.runtime.run_policy import AutoContinuePolicy, RoundContext
from scienceflow.agent.core.ports.callback_ports import (
    callback_ports_for,
    install_callback_ports,
)
from scienceflow.foundation.config.schema.settings import Config
from scienceflow.foundation.config.llm.llm_factory import build_stage_llm as _build_llm
from scienceflow.foundation.config.llm.llm_http import aclose_llm_clients
from scienceflow.runtime.task_package import find_task_package
from scienceflow.runtime.composition import build_lnr_module_graph
from scienceflow.foundation.contracts import (
    EvalContext,
    EvaluationRequest,
    EstraContext,
    EstraDecision,
)
from scienceflow.research.control.estra import EstraArchiveStore, EstraPlanner, EstraPlanRequest
from scienceflow.research.quality.evaluator.backends.command_env import (
    task_command_env,
    task_python_executable,
)
from scienceflow.research.quality.gate import (
    format_invalid_evaluator_feedback,
    legacy_score_contract_enabled,
)
from scienceflow.research.quality.assessment import CandidateAssessmentService
from scienceflow.research.quality.evaluator.adapters import (
    merge_adjudicated_stage_facts,
    merge_primary_stage_facts,
    metric_event_to_stage_facts,
)
from scienceflow.runtime.safety.tooling.resource_management.resource_policy import normalize_shell_command
from inquirycraft.tools import atomic_write
from scienceflow.research.state.knowledge.skills.catalog.paths import default_skill_library_dir
from scienceflow.research.state.knowledge.skills.catalog.registry import SkillRegistry
from scienceflow.research.solver.lnr.support.prompts import (
    build_metric_output_interpreter_prompt,
    build_metric_output_interpreter_system_prompt,
    build_metric_validity_adjudicator_prompt,
    build_metric_validity_adjudicator_system_prompt,
    build_estra_resume_prompt,
    build_estra_prompt,
    build_estra_archive_summary_prompt,
    build_stage_commit_judgment_prompt,
    build_keep_current_compact_prompt,
)
from scienceflow.research.solver.lnr.support.context_hygiene import (
    evaluate_context_hygiene_compact,
    large_code_file_touch_counts,
    large_tool_output_count,
)
from scienceflow.research.state.knowledge.memory import (
    MemoryCompactionRequest,
    MemoryService,
    ProtectedContextRequest,
    ensure_lnr_context_memory_budget,
)
from scienceflow.research.state.workspace import StageTransactionService
from scienceflow.runtime.core.stage import (
    StageLifecycleCoordinator,
    StageLifecycleEvent,
)
from scienceflow.research.solver.lnr.lifecycle.stage.metrics.metric_semantics import classify_metric_semantics
from scienceflow.research.solver.lnr.lifecycle.stage.metrics.metric_adjudication import (
    adjudicate_metric_validity,
    build_metric_validity_fields,
    parse_metric_output_interpretation_text,
    parse_metric_validity_judgment_text,
)
from scienceflow.research.solver.lnr.lifecycle.stage.metrics.metric_validity import infer_metric_validity
from scienceflow.research.solver.lnr.lifecycle.stage.records.stage_files import format_stage_files
from scienceflow.research.solver.lnr.transitions.estra_magent.join_gate import load_join_packets
from scienceflow.research.solver.lnr.transitions.estra_magent.prompt_blocks import format_magent_recommendations
from scienceflow.research.control.estra.planning.parser import parse_estra_decision
from scienceflow.research.control.estra.planning.policy import (
    axes_from_action as estra_axes_from_action,
    decision_kind as estra_decision_kind,
    derive_compact as derive_estra_compact,
)
from scienceflow.research.control.resources import LHRResourceObserver
from scienceflow.research.solver.lnr.resources.resource_advisory import (
    INLINE_RESOURCE_ADVISORY_MODE,
    build_inline_resource_advisory_prompt,
    normalize_inline_resource_advisory,
    parse_inline_resource_advisory_response,
    safe_inline_resource_advisory_boundary,
)
from scienceflow.research.solver.lnr.resources.runtime.review.decision.arbiter import (
    build_resource_arbiter_prompt,
    fallback_policy_decision,
    normalize_arbiter_decision,
    parse_arbiter_decision_text,
)
from scienceflow.research.control.admission.policy import build_resource_admission_prompt
from scienceflow.agent.factory import AgentFactory
from scienceflow.research.solver.lnr.orchestration.runtime import LnrRuntime
from scienceflow.research.solver.lnr.orchestration.runtime.services.deadline import (
    global_merge_reserve_sec,
    worker_wall_clock_budget_sec,
)
from scienceflow.research.quality.finalization.worker_outcomes import (
    multi_worker_failure_kind,
    multi_worker_stop_reason,
    worker_error_kind,
)
from scienceflow.research.solver.lnr.orchestration.runtime.execution.multi_worker import run_multi_worker
from scienceflow.research.solver.lnr.orchestration.runtime.execution.single_worker import run_single_worker
from scienceflow.research.solver.lnr.orchestration.runtime.execution.worker_environment import (
    build_worker_environment,
    format_cpu_ids,
    slice_cpu_ids,
)
from scienceflow.research.quality.finalization import metric_float
from scienceflow.research.quality.finalization.candidates.evidence import (
    apply_candidate_evidence,
    load_archived_artifact_candidates,
    load_peer_candidate_evidence,
    recover_candidate_artifact,
)
from scienceflow.research.quality.finalization.artifacts.submission_links import (
    refresh_submission_links,
)
from scienceflow.research.solver.lnr.lifecycle.stage.metrics.score_summary import (
    build_stage_performance_score_summary,
    format_score_summary_context,
    infer_stage_rows_lower_is_better,
    metric_lower_is_better_hint,
)
from scienceflow.research.solver.lnr.lifecycle.stage.records.peer_context import (
    PeerRouteEvidence,
    build_peer_route_evidence_from_csv,
)
from scienceflow.research.solver.lnr.lifecycle.snapshots.snapshot_store import SnapshotStore, StageSnapshot
from scienceflow.research.solver.lnr.lifecycle.records.stage_logs import (
    append_stage_event,
    attach_stage_interaction_handlers,
    ensure_stage_log_dir,
    reset_stage_log_dir,
    stage_log_dir,
)
from scienceflow.research.solver.lnr.lifecycle.stage.records.stage_ledger import (
    StageCard,
    append_archived_trajectory_summary,
    append_stage_event_summary,
    next_stage_id,
    normalize_stage_id,
    parse_stage_cards,
    read_ledger,
    render_stage_cards,
    stage_cards_with_overrides,
    tail_summary_from_cards,
    tail_summary_after_cards,
    validate_append_only_stage_commit,
)
from scienceflow.research.solver.lnr.orchestration.state_machine import LHR_EVENTS_JSONL, LHRStateMachineStore
from scienceflow.research.solver.lnr.lifecycle.workspace.worker_layout import ensure_lnr_worker_layout, lnr_worker_layout
from scienceflow.runtime.core.support.node_paths import find_node_log_path
from scienceflow.runtime.core.support.system_resources import parse_cpu_list
from scienceflow.runtime.observability.interaction_log import (
    attach_workspace_interaction_logger,
    close_workspace_interaction_logger,
)
from scienceflow.research.state.workspace.storage.git import (
    archive_workspace_candidate_artifact,
    auto_checkpoint_workspace_source,
    ensure_workspace_source_git,
    normalize_workspace_git_track_globs,
    workspace_source_changed,
)

logger = logging.getLogger("scienceflow")

_DETERMINISTIC_GATE_ENV = "SCIENCEFLOW_DETERMINISTIC_GATE"
_DETERMINISTIC_GATE_TIMESTAMP_ENV = "SCIENCEFLOW_DETERMINISTIC_GATE_TIMESTAMP_UTC"
_DETERMINISTIC_GATE_DEFAULT_TIMESTAMP = "2000-01-01T00:00:00Z"


def _deterministic_gate_enabled() -> bool:
    return os.environ.get(_DETERMINISTIC_GATE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _deterministic_gate_timestamp() -> str | None:
    if not _deterministic_gate_enabled():
        return None
    return (
        os.environ.get(_DETERMINISTIC_GATE_TIMESTAMP_ENV, "").strip()
        or _DETERMINISTIC_GATE_DEFAULT_TIMESTAMP
    )


def _effective_lnr_bash_timeout_sec(
    configured_sec: float,
    remaining_sec: float,
) -> float:
    """Return the per-command cap bounded by the task hard fuse."""
    configured = max(1.0, float(configured_sec or 1.0))
    remaining = max(1.0, float(remaining_sec or 1.0))
    return min(configured, remaining)


_RESOURCE_FEEDBACK_SYSTEM_PROTOCOL = """Resource feedback is runtime context, not a normal shell error.
Rules:
- PENDING/REPLAN/policy blocked: update the plan; optional standalone sleep/backoff; do not retry the same blocked GPU command unchanged.
- recommend_stop/recommend_release: resource system found inefficient work; stop, wait, or rerun only with a changed lighter/checkpointed plan.
- terminated feedback: auto mode already stopped the command; inspect artifacts/logs and replan.
- OOM/memory pressure: reduce batch, image size, model size, workers, or precision cost; CPU/light work is valid.
- RED resource mode: avoid new heavy GPU training unless clearly worth waiting for; sleep/backoff is valid.
- STOP_BOUNDARY_VIOLATION: remove physical GPU-id overrides; use runner-provided visible devices and rerun after patching.
- Resource truth: use ResourceContext/RESOURCE_FEEDBACK, not nvidia-smi or torch.cuda probes, before heavy GPU work.
- Value hint: optionally prefix valuable GPU work with SCIENCEFLOW_RESOURCE_VALUE_HINT={"expected_value_score":0.8,"near_submission_score":0.7}.
- Resource intent: prefix bash with SCIENCEFLOW_RESOURCE_INTENT=cpu_support|gpu_train|light_train|gpu_tt|readonly_cpu when the command purpose is clear; this is a hint and GPU evidence overrides CPU hints.
- Heartbeat: any command, script phase, loop, or optimization process expected to run over 2 minutes must print flushed SCIENCEFLOW_HB lines every 30-60s.
- Use SCIENCEFLOW_HB for all long phases: data preparation, search/optimization, evaluation, inference, aggregation, export, and artifact writing.
- Heartbeat format: SCIENCEFLOW_HB v=1 phase=<phase> tick=<n> elapsed_s=<sec> progress=<done>/<total_or_unknown> unit=<unit> metric=<name:value_or_na> loss=<value_or_na> artifact=<path_or_none>.
- If no metric is available, use metric=na; when a deliverable is written, print artifact=<path>.
- Bypass: do not obfuscate commands or code to evade resource classification; write normal code and let scheduling work.
- Stale context: if preflight returns stale_resource_context or newer RESOURCE_FEEDBACK, treat it as current truth and replan.
The resource manager reports constraints and observations; the main agent chooses the scientific plan using feedback, artifacts, scores, and remaining time.
"""

_LNR_MERGE_PAYLOAD_SYSTEM_BOUNDARY = """Optional reduction payload boundary. The final reducer targets three distinct defensible artifacts. During normal research, preserve materially different validated candidate artifacts when they are produced, and retain reusable prediction-level outputs under `merge_payload/` with compact metadata describing IDs, shape, class order or output semantics, and validation provenance. Do not create duplicate or cosmetically perturbed artifacts merely to reach three. For expensive inference, reuse retained predictions instead of rerunning solely for reduction; do not save model weights only for the reducer. The controller snapshots candidate artifacts and merge payloads for final cross-stage reduction.
"""


_LNR_LEDGER_SYSTEM_BOUNDARY = """LNR ledger boundary. Record experiment facts only in ordinary workspace artifacts or scratch notes such as `tmp/experiment_notes.md`, `tmp/run_notes.md`, or task-local artifacts when useful. Do not create extra planning systems, stage-like ledgers, or authoritative status files. Do not read or rewrite the hidden stage ledger unless the controller explicitly starts a stage-commit bookkeeping turn. The hidden stage ledger is append-only and controller-owned.
"""


def _append_lnr_main_agent_protocols(text: str) -> str:
    out = str(text or "").rstrip()
    for marker, block in (
        ("Resource feedback is runtime context", _RESOURCE_FEEDBACK_SYSTEM_PROTOCOL),
        ("LNR ledger boundary", _LNR_LEDGER_SYSTEM_BOUNDARY),
        ("Optional reduction payload boundary", _LNR_MERGE_PAYLOAD_SYSTEM_BOUNDARY),
    ):
        if marker not in out:
            out = (out + "\n\n" if out else "") + block.strip()
    return out + ("\n" if out else "")

LHR_STAGE_PERFORMANCE_CSV = "lhr_stage_performance.csv"
LHR_UNIFIED_ONLY_JSONL = {
    "lhr_stage_events.jsonl",
    "lhr_stage_commit_events.jsonl",
    "lhr_estra_events.jsonl",
    "lhr_estras.jsonl",
    "lhr_coordinator_events.jsonl",
    "lhr_context_events.jsonl",
}

def _validation_leakage_reason(solution_src: str, stdout_tail: str) -> str:
    src = str(solution_src or "")
    out = str(stdout_tail or "")
    src_lower = src.lower()
    out_lower = out.lower()
    concat_match = re.search(
        r"pd\s*\.\s*concat\s*\(\s*\[\s*(?:train|train_df|df_train)\s*,\s*(?:val|valid|validation|val_df|valid_df)\b",
        src,
        flags=re.IGNORECASE,
    )
    if not concat_match:
        return ""
    score_pos = src_lower.rfind("final validation score")
    if score_pos >= 0 and concat_match.start() > score_pos:
        return ""
    if "optimizing ensemble weights on validation" in out_lower:
        return "train_validation_concat_before_validation_weight_optimization"
    if "training final model" in out_lower and "validation" in out_lower and "final validation score" in out_lower:
        return "train_validation_concat_before_reported_metric"
    return ""

LHR_STAGE_PERFORMANCE_COLUMNS = [
    "row_order",
    "candidate_id",
    "worker_id",
    "worker_index",
    "worker_stage_order",
    "stage_id",
    "lineage_id",
    "node_uid",
    "parent_candidate_id",
    "parent_stage_id",
    "restored_from_candidate_id",
    "restored_from_stage",
    "restored_from_node_uid",
    "metric_value",
    "metric_name",
    "lower_is_better",
    "validation_ok",
    "validation_issue",
    "reported_val_score",
    "val_score_type",
    "selection_eligible",
    "selection_score",
    "selection_note",
    "metric_source_note",
    "metric_validity",
    "metric_validity_note",
    "metric_validity_reason_code",
    "metric_validity_confidence",
    "metric_validity_source",
    "task_profile",
    "metric_protocol",
    "train_data_used",
    "metric_eval_data",
    "artifacts_reused_from",
    "training_rows",
    "validation_rows",
    "execution_mode",
    "is_best_so_far",
    "brief",
    "why",
    "route_evidence",
    "solution_sha",
    "submission_sha",
    "source_commit_sha",
    "source_changed",
    "semantic_source_changed",
    "capture_type",
    "workspace_git_stage_id",
    "submission_snapshot",
    "artifact_path",
    "artifact_sha",
    "artifact_kind",
    "evaluator_backend",
    "evaluator_status",
    "gate_metric_validity",
    "gate_policy",
    "gate_policy_version",
    "gate_action",
    "gate_accepted",
    "gate_reason_code",
    "submission_changed",
    "candidate_ready",
    "submission_status",
    "duplicate_submission_of_stage",
    "duplicate_submission_of_snapshot_id",
    "workspace_git_ready",
    "workspace_git_message",
    "solution_run_sec",
    "elapsed_min",
    "main_llm_calls",
    "main_tokens_input",
    "main_tokens_output",
    "main_tokens_cached",
    "main_cache_rate",
    "stage_commit_llm_calls",
    "estra_llm_calls",
    "estra_decision_count_before",
    "estra_continue_count_before",
    "estra_redirect_count_before",
    "estra_switch_count_before",
    "estra_current_continue_count_before",
    "estra_current_redirect_count_before",
    "estra_stage_continue_count_before",
    "estra_stage_redirect_count_before",
    "estra_switch_stage_count_before",
    "snapshot_id",
    "snapshot_path",
    "created_at_utc",
    "route_id",
    "execution_scale",
]


class _WallClockAutoContinuePolicy(AutoContinuePolicy):
    def __init__(self, *, deadline_monotonic: float, max_text_only_retries: int = 2) -> None:
        super().__init__(max_text_only_retries=max_text_only_retries)
        self.deadline_monotonic = float(deadline_monotonic)

    def on_round_start(self, ctx: RoundContext) -> bool:
        _ = ctx
        return time.monotonic() < self.deadline_monotonic

    def on_text_only(self, ctx: RoundContext) -> tuple[bool, str | None]:
        should_continue, injection = super().on_text_only(ctx)
        if should_continue or time.monotonic() >= self.deadline_monotonic:
            return should_continue, injection
        return (
            True,
            "[LNR_CONTINUE_SEARCH] Wall-clock budget is still active. Continue "
            "the search with one valid structured tool call; a text-only response "
            "is not a stop condition.",
        )

# Export private helpers as well: responsibility modules execute the moved
# function bodies against this frozen shared semantic surface.
__all__ = (
    'annotations',
    'asyncio',
    'copy',
    'csv',
    'hashlib',
    'json',
    'logging',
    'math',
    'os',
    're',
    'shutil',
    'time',
    'Path',
    'SimpleNamespace',
    'Any',
    'Message',
    'StreamHandle',
    'ToolResult',
    'AutoContinuePolicy',
    'RoundContext',
    'callback_ports_for',
    'install_callback_ports',
    'Config',
    'aclose_llm_clients',
    'find_task_package',
    '_build_llm',
    'build_lnr_module_graph',
    'EvalContext',
    'EvaluationRequest',
    'EstraContext',
    'EstraDecision',
    'EstraArchiveStore',
    'EstraPlanner',
    'EstraPlanRequest',
    'task_command_env',
    'task_python_executable',
    'format_invalid_evaluator_feedback',
    'legacy_score_contract_enabled',
    'CandidateAssessmentService',
    'merge_adjudicated_stage_facts',
    'merge_primary_stage_facts',
    'metric_event_to_stage_facts',
    'normalize_shell_command',
    'atomic_write',
    'default_skill_library_dir',
    'SkillRegistry',
    'build_metric_output_interpreter_prompt',
    'build_metric_output_interpreter_system_prompt',
    'build_metric_validity_adjudicator_prompt',
    'build_metric_validity_adjudicator_system_prompt',
    'build_estra_resume_prompt',
    'build_estra_prompt',
    'build_estra_archive_summary_prompt',
    'build_stage_commit_judgment_prompt',
    'build_keep_current_compact_prompt',
    'evaluate_context_hygiene_compact',
    'large_code_file_touch_counts',
    'large_tool_output_count',
    'MemoryCompactionRequest',
    'MemoryService',
    'ProtectedContextRequest',
    'ensure_lnr_context_memory_budget',
    'StageTransactionService',
    'StageLifecycleCoordinator',
    'StageLifecycleEvent',
    'classify_metric_semantics',
    'adjudicate_metric_validity',
    'build_metric_validity_fields',
    'parse_metric_output_interpretation_text',
    'parse_metric_validity_judgment_text',
    'infer_metric_validity',
    'format_stage_files',
    'load_join_packets',
    'format_magent_recommendations',
    'parse_estra_decision',
    'estra_axes_from_action',
    'estra_decision_kind',
    'derive_estra_compact',
    'LHRResourceObserver',
    'INLINE_RESOURCE_ADVISORY_MODE',
    'build_inline_resource_advisory_prompt',
    'normalize_inline_resource_advisory',
    'parse_inline_resource_advisory_response',
    'safe_inline_resource_advisory_boundary',
    'build_resource_arbiter_prompt',
    'fallback_policy_decision',
    'normalize_arbiter_decision',
    'parse_arbiter_decision_text',
    'build_resource_admission_prompt',
    'AgentFactory',
    'LnrRuntime',
    'global_merge_reserve_sec',
    'worker_wall_clock_budget_sec',
    'multi_worker_failure_kind',
    'multi_worker_stop_reason',
    'worker_error_kind',
    'run_multi_worker',
    'run_single_worker',
    'build_worker_environment',
    'format_cpu_ids',
    'slice_cpu_ids',
    'metric_float',
    'apply_candidate_evidence',
    'load_archived_artifact_candidates',
    'load_peer_candidate_evidence',
    'recover_candidate_artifact',
    'refresh_submission_links',
    'build_stage_performance_score_summary',
    'format_score_summary_context',
    'infer_stage_rows_lower_is_better',
    'metric_lower_is_better_hint',
    'PeerRouteEvidence',
    'build_peer_route_evidence_from_csv',
    'SnapshotStore',
    'StageSnapshot',
    'append_stage_event',
    'attach_stage_interaction_handlers',
    'ensure_stage_log_dir',
    'reset_stage_log_dir',
    'stage_log_dir',
    'StageCard',
    'append_archived_trajectory_summary',
    'append_stage_event_summary',
    'next_stage_id',
    'normalize_stage_id',
    'parse_stage_cards',
    'read_ledger',
    'render_stage_cards',
    'stage_cards_with_overrides',
    'tail_summary_from_cards',
    'tail_summary_after_cards',
    'validate_append_only_stage_commit',
    'LHR_EVENTS_JSONL',
    'LHRStateMachineStore',
    'ensure_lnr_worker_layout',
    'lnr_worker_layout',
    'find_node_log_path',
    'parse_cpu_list',
    'attach_workspace_interaction_logger',
    'close_workspace_interaction_logger',
    'archive_workspace_candidate_artifact',
    'auto_checkpoint_workspace_source',
    'ensure_workspace_source_git',
    'normalize_workspace_git_track_globs',
    'workspace_source_changed',
    'logger',
    '_DETERMINISTIC_GATE_ENV',
    '_DETERMINISTIC_GATE_TIMESTAMP_ENV',
    '_DETERMINISTIC_GATE_DEFAULT_TIMESTAMP',
    '_deterministic_gate_enabled',
    '_deterministic_gate_timestamp',
    '_effective_lnr_bash_timeout_sec',
    '_RESOURCE_FEEDBACK_SYSTEM_PROTOCOL',
    '_LNR_MERGE_PAYLOAD_SYSTEM_BOUNDARY',
    '_LNR_LEDGER_SYSTEM_BOUNDARY',
    '_append_lnr_main_agent_protocols',
    'LHR_STAGE_PERFORMANCE_CSV',
    'LHR_UNIFIED_ONLY_JSONL',
    '_validation_leakage_reason',
    'LHR_STAGE_PERFORMANCE_COLUMNS',
    '_WallClockAutoContinuePolicy',
)
