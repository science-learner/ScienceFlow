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

"""Canonical resource observation and compatibility coordination surface.

Resource state, scheduling, leases, policy, and effects are owned by the
``scienceflow.research.control.resources`` components. This coordinator adapts the
historic observer callback protocol without making the old module path an
implementation dependency.
"""

from __future__ import annotations

__scienceflow_canonical_backend__ = True

import asyncio
import hashlib
import inspect
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from scienceflow.runtime.safety.tooling.resource_management.resource_policy import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_LIGHT_CPU,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_READONLY_CPU,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
)
from scienceflow.research.control.resources import (
    GPUQueueConfig,
    ResourceManagementService,
    ResourceReviewEvent,
    ResourceReviewEventKind,
    ResourceReviewMachine,
    ResourceEffectCommand,
    ResourceEffectKind,
    ResourceRegistry,
    project_gpu_share_decision_facts,
)
from scienceflow.research.control.execution_value.adapters import LnrExecutionValueShadow
from scienceflow.research.control.admission import (
    AdmissionDecisionRouter,
    AdmissionPolicyService,
    AdmissionReplayArchive,
)
from scienceflow.research.control.admission.policy import (
    apply_admission_decision,
    parse_admission_decision_text,
)
from scienceflow.research.solver.lnr.resources.runtime.review.decision.arbiter import (
    TERMINATING_ACTIONS,
    enforce_proposal_action_allowlist,
    enforce_repeated_stall_escalation,
    fallback_policy_decision,
    main_agent_feedback,
    normalize_arbiter_decision,
    parse_arbiter_decision_text,
)
from scienceflow.research.solver.lnr.resources.runtime.review.decision.arbiter_gate import (
    enforce_arbiter_kill_gate,
    proposal_advisory_fact_conflicts,
)
from scienceflow.research.solver.lnr.resources.runtime.review.decision.kill_intent import (
    build_kill_intent_snapshot,
    revalidate_kill_intent as evaluate_kill_intent_revalidation,
)
from scienceflow.research.solver.lnr.resources.runtime.control.admission.control_profile import (
    ResourceControlProfile,
    apply_control_profile_to_decision_delay,
    build_resource_control_profile,
    cap_float,
    cap_int,
)
from scienceflow.research.solver.lnr.resources.runtime.review.evidence.progress import classify_progress_signal
from scienceflow.runtime.safety.resource.lifecycle.process_lifecycle import classify_process_lifecycle
from scienceflow.research.solver.lnr.resources.runtime.execution.process.process_liveness import build_process_liveness
from scienceflow.research.solver.lnr.resources.runtime.execution.process.proc_inspect import read_proc_info
from scienceflow.research.solver.lnr.resources.runtime.execution.state.utilization import process_tree_gpu_placement_snapshot
from scienceflow.research.solver.lnr.resources.runtime.execution.process.quick_probe import classify_quick_probe_command
from scienceflow.research.solver.lnr.resources.runtime.review.evidence.state_generation import build_resource_state_generation
from scienceflow.research.solver.lnr.resources.runtime.review.decision.execution_facts import build_execution_facts
from scienceflow.research.solver.lnr.resources.runtime.review.decision.efficiency import assess_resource_efficiency
from scienceflow.research.solver.lnr.resources.runtime.review.evidence.research_cadence import (
    build_research_cadence_facts,
    is_comparable_live_metric,
    normalize_execution_scale,
    normalize_route_key,
    normalize_validation_protocol,
    route_metric_evidence,
    validation_protocols_comparable,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.metric_history import (
    metric_history_line_from_progress_signals,
    metric_history_text as compact_metric_history_text,
    update_metric_history_lines,
)
from scienceflow.research.solver.lnr.resources.runtime.control.gpu.gpu_sharing import (
    build_gpu_share_config,
    classify_cpu_pressure,
    evaluate_admission_share_trial,
    evaluate_share_phase_a,
    gpu_memory_summary,
)
from scienceflow.research.solver.lnr.resources.runtime.review.evidence.lease_suspect import (
    LeaseSuspectThresholds,
    active_work_counterevidence,
    classify_active_lease_suspect,
    classify_deliverable_validity,
    classify_route_viability,
)
from scienceflow.research.solver.lnr.transitions.estra_magent.coordinator import ESTRAMagentConfig
from scienceflow.research.solver.lnr.transitions.estra_magent.events import (
    MAGENT_FORK_CONSIDERED,
    MAGENT_JOIN_PACKET_READY,
    MAGENT_SIDECAR_STARTED,
    MAGENT_TRAIN_OBSERVED,
    fork_considered_payload,
    join_packet_ready_payload,
    sidecar_started_payload,
    train_observed_payload,
)
from scienceflow.research.solver.lnr.transitions.estra_magent.join_gate import write_join_packet
from scienceflow.runtime.safety.resource.lifecycle.completion import deliverable_completion_state
from scienceflow.runtime.safety.resource import (
    KILL as RESOURCE_REVIEW_KILL,
    NO_ACTION as RESOURCE_REVIEW_NO_ACTION,
    TIMEBOX as RESOURCE_REVIEW_TIMEBOX,
    PROGRESS_WINDOW,
    RESOURCE_PRESSURE,
    ROUTE_VALUE,
    STALL,
    TIMEBOX_EXPIRED,
    ResourceReviewConfig,
    ResourceReviewState,
    build_review_signal,
    compute_timebox_sec,
    new_review_state,
    next_review_boundary,
    normalize_clear_on,
    normalize_value_review_outcome,
)
from scienceflow.research.solver.lnr.resources.runtime.control.policy.post_feedback_actions import (
    decorate_post_feedback_action_after_backoff,
)
from scienceflow.research.control.resources.runtime.feedback import resource_feedback_text
from scienceflow.research.solver.lnr.resources.runtime.execution.process.sidecar import run_cpu_sidecar_backfill
from scienceflow.research.solver.lnr.resources.runtime.control.gpu.soft_gpu_gate import (
    feedback_guard_is_hard_constraint,
    gpu_store_snapshot_has_lease,
    gpu_store_snapshot_has_pressure,
    normalize_gpu_ids,
    reconcile_free_gpu_pressure_blocked,
    resource_class_can_use_soft_gpu_gate,
)
from scienceflow.research.solver.lnr.resources.runtime.control.policy.startup_trial import (
    classify_preflight_block,
    startup_policy_trial_first,
    trial_windows_for_profile,
)
from scienceflow.research.solver.lnr.lifecycle.stage.metrics.score_summary import (
    build_stage_performance_score_summary,
    metric_lower_is_better_hint,
)
from scienceflow.research.solver.lnr.resources.runtime.control.policy.source_hints import (
    ResourceSourceHint,
    detect_resource_source_hint,
)
from scienceflow.research.solver.lnr.orchestration.state_machine import LHRStateMachineStore
from scienceflow.research.solver.lnr.resources.runtime.execution.process.jobs import (
    ResourceJob,
    TRACKABLE_CLASSES,
    TRAIN_CLASSES,
    TT_ALLOWED_AFTER_TIMEOUT,
)

# Preserve the historical event value during the Safety namespace migration.
# This is a compatibility token, not the name of an internal module.
_SAFETY_HEARTBEAT_SOURCE = "safety_heartbeat"

# Export private helpers as well: bridge modules execute mechanically moved
# callback bodies against this frozen shared semantic surface.
__all__ = (
    'AdmissionDecisionRouter',
    'AdmissionPolicyService',
    'AdmissionReplayArchive',
    'Any',
    'ESTRAMagentConfig',
    'GPUQueueConfig',
    'LHRStateMachineStore',
    'LeaseSuspectThresholds',
    'LnrExecutionValueShadow',
    'MAGENT_FORK_CONSIDERED',
    'MAGENT_JOIN_PACKET_READY',
    'MAGENT_SIDECAR_STARTED',
    'MAGENT_TRAIN_OBSERVED',
    'PROGRESS_WINDOW',
    'Path',
    'RESOURCE_GPU_FEATURE_EXTRACT',
    'RESOURCE_GPU_LIGHT_TRAIN',
    'RESOURCE_GPU_TT_LIGHT',
    'RESOURCE_HEAVY_CPU_CANDIDATE',
    'RESOURCE_HEAVY_GPU_CANDIDATE',
    'RESOURCE_HEAVY_GPU_TRAIN',
    'RESOURCE_LIGHT_CPU',
    'RESOURCE_LIGHT_GPU_PROBE',
    'RESOURCE_PRESSURE',
    'RESOURCE_PURE_TT_CPU',
    'RESOURCE_READONLY_CPU',
    'RESOURCE_REVIEW_KILL',
    'RESOURCE_REVIEW_NO_ACTION',
    'RESOURCE_REVIEW_TIMEBOX',
    'RESOURCE_UNKNOWN_EXEC',
    'RESOURCE_UNKNOWN_GPU_EXEC',
    'ROUTE_VALUE',
    'ResourceControlProfile',
    'ResourceEffectCommand',
    'ResourceEffectKind',
    'ResourceJob',
    'ResourceManagementService',
    'ResourceRegistry',
    'ResourceReviewConfig',
    'ResourceReviewEvent',
    'ResourceReviewEventKind',
    'ResourceReviewMachine',
    'ResourceReviewState',
    'ResourceSourceHint',
    'STALL',
    'TERMINATING_ACTIONS',
    'TIMEBOX_EXPIRED',
    'TRACKABLE_CLASSES',
    'TRAIN_CLASSES',
    'TT_ALLOWED_AFTER_TIMEOUT',
    '_SAFETY_HEARTBEAT_SOURCE',
    'active_work_counterevidence',
    'apply_admission_decision',
    'apply_control_profile_to_decision_delay',
    'assess_resource_efficiency',
    'asyncio',
    'build_execution_facts',
    'build_gpu_share_config',
    'build_kill_intent_snapshot',
    'build_process_liveness',
    'build_research_cadence_facts',
    'build_resource_control_profile',
    'build_resource_state_generation',
    'build_review_signal',
    'build_stage_performance_score_summary',
    'cap_float',
    'cap_int',
    'classify_active_lease_suspect',
    'classify_cpu_pressure',
    'classify_deliverable_validity',
    'classify_preflight_block',
    'classify_process_lifecycle',
    'classify_progress_signal',
    'classify_quick_probe_command',
    'classify_route_viability',
    'compact_metric_history_text',
    'compute_timebox_sec',
    'decorate_post_feedback_action_after_backoff',
    'deliverable_completion_state',
    'detect_resource_source_hint',
    'enforce_arbiter_kill_gate',
    'enforce_proposal_action_allowlist',
    'enforce_repeated_stall_escalation',
    'evaluate_admission_share_trial',
    'evaluate_kill_intent_revalidation',
    'evaluate_share_phase_a',
    'fallback_policy_decision',
    'feedback_guard_is_hard_constraint',
    'fork_considered_payload',
    'gpu_memory_summary',
    'gpu_store_snapshot_has_lease',
    'gpu_store_snapshot_has_pressure',
    'hashlib',
    'inspect',
    'is_comparable_live_metric',
    'join_packet_ready_payload',
    'json',
    'main_agent_feedback',
    'metric_history_line_from_progress_signals',
    'metric_lower_is_better_hint',
    'new_review_state',
    'next_review_boundary',
    'normalize_arbiter_decision',
    'normalize_clear_on',
    'normalize_execution_scale',
    'normalize_gpu_ids',
    'normalize_route_key',
    'normalize_validation_protocol',
    'normalize_value_review_outcome',
    'parse_admission_decision_text',
    'parse_arbiter_decision_text',
    'process_tree_gpu_placement_snapshot',
    'project_gpu_share_decision_facts',
    'proposal_advisory_fact_conflicts',
    're',
    'read_proc_info',
    'reconcile_free_gpu_pressure_blocked',
    'replace',
    'resource_class_can_use_soft_gpu_gate',
    'resource_feedback_text',
    'route_metric_evidence',
    'run_cpu_sidecar_backfill',
    'sidecar_started_payload',
    'startup_policy_trial_first',
    'time',
    'train_observed_payload',
    'trial_windows_for_profile',
    'update_metric_history_lines',
    'validation_protocols_comparable',
    'write_join_packet',
)
