# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Extracted coordinator component with an explicit dependency surface."""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    LHR_STAGE_PERFORMANCE_CSV,
    _deterministic_gate_enabled,
    _deterministic_gate_timestamp,
    build_stage_performance_score_summary,
    format_magent_recommendations,
    format_score_summary_context,
    load_join_packets,
    logger,
    os,
    parse_cpu_list,
    re,
)


@staticmethod
def _compact_peer_method(text: str, *, max_chars: int = 180) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    if not out:
        return "method not recorded"
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _magent_recommendations_for_resource_context(self) -> str:
    lhr_cfg = getattr(self, "lhr", None)
    if not bool(getattr(lhr_cfg, "estra_magent_enabled", False)):
        return ""
    if not bool(getattr(lhr_cfg, "estra_magent_join_inject_resource_context", True)):
        return ""
    runtime = getattr(
        getattr(self, "resource_observer", None), "resource_runtime", None
    )
    resource_dir = getattr(runtime, "resource_dir", None)
    if resource_dir is None:
        return ""
    try:
        packets = load_join_packets(
            resource_dir, parent_worker_id=self.worker_id or "W00", max_packets=3
        )
        return format_magent_recommendations(packets, max_items=3, max_chars=1200)
    except Exception:
        logger.debug("[lnr] estra magent recommendations failed", exc_info=True)
        return ""


def _stage_payloads_for_score_summary(self) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for sid, snap in sorted(getattr(self, "stage_snapshots", {}).items()):
        source = (
            snap.source_event
            if isinstance(getattr(snap, "source_event", None), dict)
            else {}
        )
        payloads.append({"stage_id": sid, "metric_event": source})
    return payloads


def _score_summary_for_prompt(self) -> dict[str, Any]:
    try:
        return build_stage_performance_score_summary(
            self.global_log_dir / LHR_STAGE_PERFORMANCE_CSV,
            current_worker_id=self._worker_uid_prefix(),
            fallback_stage_payloads=self._stage_payloads_for_score_summary(),
        )
    except Exception:
        logger.debug("[lnr] stage-performance score summary failed", exc_info=True)
        return {}


def _allocated_compute_context_lines(self) -> list[str]:
    exec_cfg = getattr(getattr(self, "cfg", None), "exec", None)
    worker_env = (
        self.worker_extra_env
        if isinstance(getattr(self, "worker_extra_env", None), dict)
        else {}
    )
    worker_cpu = (
        str(worker_env.get("SCIENCEFLOW_WORKER_CPU_LIST") or "").strip()
        or str(worker_env.get("SCIENCEFLOW_CPU_LIST") or "").strip()
        or str(worker_env.get("_SCIENCEFLOW_CPU_SET") or "").strip()
        or os.environ.get("SCIENCEFLOW_WORKER_CPU_LIST", "").strip()
        or os.environ.get("_SCIENCEFLOW_CPU_SET", "").strip()
    )
    task_cpu = (
        str(worker_env.get("SCIENCEFLOW_TASK_CPU_LIST") or "").strip()
        or os.environ.get("SCIENCEFLOW_TASK_CPU_LIST", "").strip()
    )
    cfg_cpu = str(getattr(exec_cfg, "cpu_list", "") or "").strip()
    cpu_text = worker_cpu or task_cpu or cfg_cpu
    cpu_count = 0
    if cpu_text:
        try:
            cpu_count = len(parse_cpu_list(cpu_text))
        except Exception:
            logger.debug(
                "[lnr] allocated compute CPU parse failed: %s", cpu_text, exc_info=True
            )
    display_cpu_text = cpu_text
    display_task_cpu = task_cpu
    if _deterministic_gate_enabled() and cpu_count > 0:
        logical_start = (
            int(getattr(self, "worker_index", 0) or 0) * cpu_count if worker_cpu else 0
        )
        display_cpu_text = f"{logical_start}-{logical_start + cpu_count - 1}"
        if task_cpu:
            try:
                task_cpu_count = len(parse_cpu_list(task_cpu))
            except Exception:
                task_cpu_count = 0
            if task_cpu_count > 0:
                display_task_cpu = f"0-{task_cpu_count - 1}"
    omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()

    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    task_gpu = os.environ.get("SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL", "").strip()
    cfg_gpu = str(getattr(exec_cfg, "gpu_list", "") or "").strip()
    gpu_text = visible_gpu or task_gpu or cfg_gpu

    if not any((cpu_text, task_cpu, cfg_cpu, omp_threads, gpu_text, task_gpu, cfg_gpu)):
        return []

    lines = ["allocated_compute:"]
    worker_id = self._worker_uid_prefix()
    if worker_id:
        lines.append(f"  - worker_id: {worker_id}")
    if display_cpu_text:
        label = "worker_cpu_list" if worker_cpu else "task_cpu_list"
        lines.append(f"  - {label}: {display_cpu_text}")
        if cpu_count > 0:
            lines.append(f"  - worker_cpu_count: {cpu_count}")
    if display_task_cpu and display_task_cpu != display_cpu_text:
        lines.append(f"  - task_cpu_list: {display_task_cpu}")
    if omp_threads:
        lines.append(f"  - thread_env: OMP_NUM_THREADS={omp_threads}")
    if gpu_text:
        label = "visible_gpu_list" if visible_gpu else "task_gpu_list"
        if gpu_text == "-1":
            lines.append(f"  - {label}: none")
        else:
            lines.append(f"  - {label}: {gpu_text}")
            if visible_gpu:
                lines.append(
                    "  - cuda_index_note: use cuda:0 for the first visible assigned GPU"
                )
    if cfg_gpu and cfg_gpu != gpu_text:
        lines.append(f"  - configured_gpu_list: {cfg_gpu}")
    lines.append(
        "  - expectation: scale validated routes to use assigned CPU/GPU; if staying resource-light, state why"
    )
    return lines


def _render_resource_context_prompt(
    self,
    *,
    allocation_lines,
    context_version,
    observed_at_utc,
    pressure_generation,
    mode,
    policy_gates,
    queue_timeouts,
    admission_deferred,
    score_summary,
    has_score_context,
    pressure_rows,
    active,
    pending,
    allowed,
    blocked,
    last_boundary_event,
    cooldown,
    eta,
    last_feedback,
    magent_block,
) -> str:
    lines = [
        *allocation_lines,
        f"resource_context_version: {context_version}",
        f"observed_at_utc: {observed_at_utc or ''}",
        "pressure_scope: gpu",
        f"pressure_generation: {pressure_generation}",
        f"resource_mode: {mode}",
        f"policy_gates: {policy_gates}",
        f"queue_timeouts: {queue_timeouts}",
        f"admission_pending_or_replan: {admission_deferred}",
    ]
    if has_score_context:
        lines.extend(format_score_summary_context(score_summary))
    if pressure_rows:
        lines.append("gpu_pressure:")
        for row in pressure_rows[:4]:
            gpu_id = str(row.get("gpu_id") or "")
            row_mode = str(row.get("mode") or "")
            qtc = int(row.get("queue_timeout_count") or 0)
            cd = max(0.0, float(row.get("cooldown_remaining_sec") or 0.0))
            reason = str(row.get("last_reason") or row.get("last_event") or "")
            lines.append(
                f"  - gpu_id: {gpu_id}; mode: {row_mode}; queue_timeout_count: {qtc}; cooldown_sec: {cd:.0f}; reason: {reason}"
            )
    if active:
        lines.append("active_leases:")
        for row in active[:4]:
            lines.append(
                f"  - worker_id: {row.get('worker_id') or ''}; class: {row.get('resource_class') or ''}; gpu_ids: {','.join((str(x) for x in row.get('gpu_ids') or []))}; slot_weight: {row.get('slot_weight') or ''}"
            )
    if pending:
        lines.append("pending_gpu_jobs:")
        for row in pending[:4]:
            lines.append(
                f"  - worker_id: {row.get('worker_id') or ''}; class: {row.get('resource_class') or ''}; gpu_ids: {','.join((str(x) for x in row.get('gpu_ids') or []))}"
            )
    lines.append("allowed_classes:")
    lines.extend((f"  - {x}" for x in allowed[:8]))
    if blocked:
        lines.append("blocked_classes:")
        lines.extend((f"  - {x}" for x in blocked[:8]))
    if last_boundary_event:
        actual = (
            ",".join(
                (
                    str(x)
                    for x in last_boundary_event.get("actual_gpu_ids") or []
                    if str(x).strip()
                )
            )
            or "unknown"
        )
        allowed_gpu = (
            ",".join(
                (
                    str(x)
                    for x in last_boundary_event.get("allowed_gpu_ids") or []
                    if str(x).strip()
                )
            )
            or "unknown"
        )
        lines.append("recent_boundary_event:")
        lines.append(
            "  - action: {action}; reason: {reason}; actual_gpu_ids: {actual}; allowed_gpu_ids: {allowed}; command_digest: {digest}".format(
                action=str(last_boundary_event.get("post_feedback_action") or ""),
                reason=str(last_boundary_event.get("reason") or ""),
                actual=actual,
                allowed=allowed_gpu,
                digest=str(last_boundary_event.get("command_digest") or ""),
            )
        )
    if cooldown > 0:
        lines.append(f"cooldown_sec: {cooldown:.0f}")
    if eta > 0:
        lines.append(f"eta_next_train_sec: {eta:.0f}")
        lines.append(
            f"eta_confidence: {str((last_feedback or {}).get('eta_confidence') or 'low')}"
        )
        eta_source = str((last_feedback or {}).get("eta_source") or "")
        if eta_source:
            lines.append(f"eta_source: {eta_source}")
        if (last_feedback or {}).get("runtime_history_count") is not None:
            lines.append(
                f"runtime_history_count: {last_feedback.get('runtime_history_count')}"
            )
    if last_feedback and last_feedback.get("admission_priority_score") is not None:
        lines.append(
            f"admission_priority_score: {last_feedback.get('admission_priority_score')}"
        )
        if last_feedback.get("expected_value_score") is not None:
            lines.append(
                f"expected_value_score: {last_feedback.get('expected_value_score')}"
            )
        if last_feedback.get("near_submission_score") is not None:
            lines.append(
                f"near_submission_score: {last_feedback.get('near_submission_score')}"
            )
    lines.append(
        "resource_feedback_contract: facts_and_constraint_logic_only; main_agent_chooses_research_route"
    )
    if magent_block:
        lines.append(magent_block)
    lines.append(
        "long_command_heartbeat: for any command, script phase, loop, or optimization process expected to run >2 minutes, print one flushed SCIENCEFLOW_HB stdout line every 30-60s"
    )
    lines.append(
        "heartbeat_scope: cover data prep, search/optimization, evaluation, inference, aggregation, export, and artifact writing; use metric=na when no metric exists"
    )
    lines.append(
        "heartbeat_format: SCIENCEFLOW_HB v=1 phase=<phase> tick=<n> elapsed_s=<sec> progress=<done>/<total_or_unknown> unit=<unit> metric=<name:value_or_na> loss=<value_or_na> artifact=<path_or_none>"
    )
    lines.append(
        "heartbeat_artifact: when a deliverable is written, print artifact=<path> in the heartbeat"
    )
    lines.append(
        "gpu_start_decision_source: use ResourceContext/RESOURCE_FEEDBACK lease status; nvidia-smi and torch.cuda probes are advisory"
    )
    lines.append(
        "resource_intent_hint: prefix clear bash commands with SCIENCEFLOW_RESOURCE_INTENT=cpu_support|gpu_train|light_train|gpu_tt|readonly_cpu; GPU evidence overrides CPU hints"
    )
    lines.append(
        "stale_resource_context_behavior: if preflight returns new RESOURCE_FEEDBACK, treat it as truth and replan"
    )
    return "\n".join(lines)


def _merge_global_resource_state(
    self,
    *,
    pressure_rows: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    active: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    pressure_generation = 0
    runtime = getattr(
        getattr(self, "resource_observer", None), "resource_runtime", None
    )
    if runtime is not None:
        try:
            pressure_snapshot = runtime.pressure_store.snapshot()
            pressure_generation = int(pressure_snapshot.get("generation") or 0)
            global_pressure_rows = [
                dict(row)
                for row in (pressure_snapshot.get("gpus") or {}).values()
                if isinstance(row, dict)
            ]
            if global_pressure_rows:
                by_gpu = {
                    str(row.get("gpu_id") or idx): row
                    for (idx, row) in enumerate(pressure_rows)
                }
                for row in global_pressure_rows:
                    by_gpu[str(row.get("gpu_id") or len(by_gpu))] = row
                pressure_rows = list(by_gpu.values())
        except Exception:
            logger.debug(
                "[lnr] global resource pressure snapshot failed", exc_info=True
            )
        try:
            active_snapshot = runtime.gpu_store.snapshot_active()
            leases = (
                active_snapshot.get("leases")
                if isinstance(active_snapshot.get("leases"), dict)
                else {}
            )
            waiters = (
                active_snapshot.get("waiters")
                if isinstance(active_snapshot.get("waiters"), dict)
                else {}
            )
            active_by_job = {
                str(row.get("job_id") or idx): row for (idx, row) in enumerate(active)
            }
            for job_id, raw in leases.items():
                if not isinstance(raw, dict):
                    continue
                meta = (
                    raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                )
                active_by_job[str(job_id)] = {
                    "job_id": str(job_id),
                    "worker_id": str(raw.get("worker_id") or ""),
                    "resource_class": str(
                        meta.get("policy_resource_class")
                        or meta.get("resource_class")
                        or ""
                    ),
                    "gpu_ids": [
                        str(x) for x in raw.get("gpu_ids") or [] if str(x).strip()
                    ],
                    "slot_weight": meta.get("slot_weight"),
                }
            active = list(active_by_job.values())
            pending_by_job = {
                str(row.get("job_id") or idx): row for (idx, row) in enumerate(pending)
            }
            for job_id, raw in waiters.items():
                if not isinstance(raw, dict):
                    continue
                pending_by_job[str(job_id)] = {
                    "job_id": str(job_id),
                    "worker_id": str(raw.get("worker_id") or ""),
                    "resource_class": str(raw.get("resource_class") or ""),
                    "gpu_ids": [
                        str(x) for x in raw.get("gpu_ids") or [] if str(x).strip()
                    ],
                    "queue_position": raw.get("queue_position"),
                    "slot_weight": raw.get("slot_weight"),
                }
            pending = list(pending_by_job.values())
        except Exception:
            logger.debug("[lnr] global resource lease snapshot failed", exc_info=True)
    return pressure_rows, pending, active, pressure_generation


def _resource_prompt_policy(
    self,
    *,
    state: dict[str, Any],
    pressure_rows: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    policy_gates: int,
    queue_timeouts: int,
    admission_deferred: int,
    last_feedback: dict[str, Any] | None,
) -> tuple[str, list[str], list[str], float, float, str]:
    modes = {str(row.get("mode") or "").upper() for row in pressure_rows}
    feedback_mode = str((last_feedback or {}).get("resource_mode") or "").upper()
    feedback_status = str((last_feedback or {}).get("feedback_status") or "").upper()
    if (
        "RED" in modes
        or feedback_mode == "RED"
        or policy_gates > 0
        or (queue_timeouts > 0)
    ):
        mode = "RED"
    elif (
        pending
        or admission_deferred > 0
        or feedback_mode == "YELLOW"
        or (feedback_status in {"DEFERRED", "PENDING"})
    ):
        mode = "YELLOW"
    else:
        mode = "GREEN"
    allowed = []
    if last_feedback and isinstance(last_feedback.get("allowed_classes"), list):
        allowed = [
            str(x) for x in last_feedback.get("allowed_classes") if str(x).strip()
        ]
    if not allowed:
        if mode == "RED":
            allowed = ["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu"]
        elif mode == "YELLOW":
            allowed = [
                "pure_tt_cpu",
                "gpu_tt_light",
                "readonly_cpu",
                "light_cpu",
                "one_deferred_train",
            ]
        else:
            allowed = [
                "heavy_gpu_train",
                "gpu_feature_extract",
                "gpu_tt_light",
                "pure_tt_cpu",
            ]
    blocked = []
    blocked_class = str((last_feedback or {}).get("blocked_class") or "").strip()
    if mode == "RED":
        blocked = ["heavy_gpu_train", "heavy_gpu_candidate", "unknown_gpu_exec"]
    elif mode == "YELLOW":
        blocked = [
            "duplicate_failed_digest",
            "unknown_gpu_exec",
            "low_value_repeat_train",
        ]
        if (
            feedback_status in {"DEFERRED", "PENDING", "REPLAN", "DENIED_REPLAN"}
            and blocked_class
        ):
            blocked.insert(0, blocked_class)
            if blocked_class == "heavy_gpu_candidate":
                blocked.insert(1, "heavy_gpu_train")
    cooldown = max(
        [
            float(row.get("cooldown_remaining_sec") or row.get("cooldown_sec") or 0.0)
            for row in pressure_rows
        ]
        or [0.0]
    )
    eta = cooldown + 300.0 if mode == "RED" else 0.0
    if last_feedback and last_feedback.get("eta_next_train_sec") is not None:
        try:
            eta = max(0.0, float(last_feedback.get("eta_next_train_sec") or 0.0))
        except (TypeError, ValueError):
            pass
    observed_at_utc = (
        _deterministic_gate_timestamp()
        if _deterministic_gate_enabled()
        else str(state.get("generated_at_utc") or "")
    )
    return mode, allowed, blocked, cooldown, eta, observed_at_utc


def _resource_context_for_prompt(self) -> str:
    lhr_cfg = getattr(self, "lhr", None)
    if not bool(getattr(lhr_cfg, "resource_context_prompt_enabled", True)):
        return ""
    magent_block = self._magent_recommendations_for_resource_context()
    allocation_lines = self._allocated_compute_context_lines()
    state_machine = getattr(self, "state_machine", None)
    if state_machine is None:
        parts = [*allocation_lines]
        if magent_block:
            parts.append(magent_block)
        return "\n".join(parts).strip()
    try:
        state = state_machine._base_state()
    except Exception:
        logger.debug("[lnr] resource context snapshot failed", exc_info=True)
        parts = [*allocation_lines]
        if magent_block:
            parts.append(magent_block)
        return "\n".join(parts).strip()
    context_version = int(state.get("event_count") or 0)
    pressure_rows = [
        x
        for x in state.get("resource_gpu_pressure_states") or []
        if isinstance(x, dict)
    ]
    pending = [
        x for x in state.get("resource_pending_gpu_jobs") or [] if isinstance(x, dict)
    ]
    active = [
        x for x in state.get("resource_active_leases") or [] if isinstance(x, dict)
    ]
    pressure_rows, pending, active, pressure_generation = _merge_global_resource_state(
        self, pressure_rows=pressure_rows, pending=pending, active=active
    )
    try:
        if self.resource_observer is not None:
            self.resource_observer.set_planning_resource_context_version(
                context_version, pressure_generation=pressure_generation
            )
    except Exception:
        logger.debug("[lnr] resource context version sync failed", exc_info=True)
    policy_gates = int(state.get("resource_policy_gates") or 0)
    queue_timeouts = int(state.get("resource_gpu_queue_timeouts") or 0)
    admission_deferred = int(state.get("resource_admission_deferred") or 0)
    resource_last_rows = [
        x for x in state.get("resource_last_events") or [] if isinstance(x, dict)
    ]
    score_summary = self._score_summary_for_prompt()
    has_score_context = bool(
        (score_summary.get("best_score") or {})
        or score_summary.get("cheap_signal_available")
    )
    last_feedback = None
    last_boundary_event = None
    for row in reversed(resource_last_rows):
        action = str(row.get("post_feedback_action") or "").lower()
        reason = str(row.get("reason") or "").lower()
        if last_boundary_event is None and (
            action == "stop_boundary_violation" or "boundary_violation" in reason
        ):
            last_boundary_event = row
        if (
            row.get("feedback_status")
            or row.get("resource_mode")
            or row.get("allowed_classes")
        ):
            last_feedback = row
            if last_boundary_event is not None:
                break
    if (
        not pressure_rows
        and (not pending)
        and (not active)
        and (policy_gates <= 0)
        and (queue_timeouts <= 0)
        and (admission_deferred <= 0)
        and (last_feedback is None)
        and (last_boundary_event is None)
        and (not magent_block)
        and (not has_score_context)
        and (not allocation_lines)
    ):
        return ""
    mode, allowed, blocked, cooldown, eta, observed_at_utc = _resource_prompt_policy(
        self,
        state=state,
        pressure_rows=pressure_rows,
        pending=pending,
        policy_gates=policy_gates,
        queue_timeouts=queue_timeouts,
        admission_deferred=admission_deferred,
        last_feedback=last_feedback,
    )
    return self._render_resource_context_prompt(
        allocation_lines=allocation_lines,
        context_version=context_version,
        observed_at_utc=observed_at_utc,
        pressure_generation=pressure_generation,
        mode=mode,
        policy_gates=policy_gates,
        queue_timeouts=queue_timeouts,
        admission_deferred=admission_deferred,
        score_summary=score_summary,
        has_score_context=has_score_context,
        pressure_rows=pressure_rows,
        active=active,
        pending=pending,
        allowed=allowed,
        blocked=blocked,
        last_boundary_event=last_boundary_event,
        cooldown=cooldown,
        eta=eta,
        last_feedback=last_feedback,
        magent_block=magent_block,
    )
