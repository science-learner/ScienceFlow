# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: agent construction, runtime context, model routing, and resume.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    Path,
    _append_lnr_main_agent_protocols,
    _deterministic_gate_enabled,
    _effective_lnr_bash_timeout_sec,
    _WallClockAutoContinuePolicy,
    append_archived_trajectory_summary,
    callback_ports_for,
    copy,
    install_callback_ports,
    legacy_score_contract_enabled,
    logger,
    math,
    normalize_workspace_git_track_globs,
    time,
)


def _make_agent(self, *, load_existing_memory: bool) -> Any:
    agent = _create_main_agent(self, load_existing_memory=load_existing_memory)
    _configure_agent_workspace_surface(self, agent)
    _configure_agent_bash(self, agent)
    _configure_agent_protocol(self, agent)
    _configure_agent_callbacks(self, agent)
    return agent


def _create_main_agent(self, *, load_existing_memory: bool) -> Any:
    policy = _WallClockAutoContinuePolicy(
        deadline_monotonic=self.deadline, max_text_only_retries=0
    )
    node_id = "lnr" if not self.worker_id else f"lnr:{self.worker_id}"
    process_id = getattr(self.cfg, "exp_id", "") or "lhr"
    if self.worker_id:
        process_id = f"{process_id}:{self.worker_id}"
    llm_stage_override = self._worker_llm_stage_override()
    hook = self.orchestrator.make_llm_call_tracer(
        node_id=node_id,
        process_id=process_id,
        detail_prefix="mode=lnr;role=main_agent"
        + (f";worker={self.worker_id}" if self.worker_id else ""),
    )
    agent_extra_env = dict(self.worker_extra_env)
    agent_extra_env.update(self._task_runtime_extra_env())
    agent = self._agent_factory_service().create(
        "main_agent",
        correlation={"worker_id": getattr(self, "worker_id", "") or "W00"},
        task_description=None,
        run_policy=policy,
        max_steps_override=max(1, int(self.lhr.max_steps or 1)),
        append_repl_system_prompt=True,
        pin_task_description=False,
        on_llm_call=hook,
        memory_dir_override=self.memory_dir,
        load_existing_memory=load_existing_memory,
        memory_agent_name="ScienceAgent",
        teleport_mode="off",
        repl_bash_write_mode=True,
        stable_system_prompt=True,
        pin_environment_context=False,
        code_organization_hint=self._code_organization_hint(),
        workspace_git_enabled=bool(getattr(self.lhr, "workspace_git_enabled", True)),
        workspace_git_track_globs=list(
            normalize_workspace_git_track_globs(
                getattr(self.lhr, "workspace_git_track_globs", None)
            ),
        ),
        workspace_git_auto_review=bool(
            getattr(self.lhr, "workspace_git_auto_review", False)
        ),
        # LNR owns source checkpoints at metric-backed stage boundaries, with
        # ledgers under worker logs/checkpoints. Disabling the agent-level
        # per-tool checkpoint avoids a second workspace/.scienceflow_checkpoints ledger.
        workspace_git_auto_checkpoint=False,
        bash_max_output_chars_override=int(
            getattr(self.cfg, "repl_bash_max_output_chars", 8000) or 8000
        ),
        bash_max_stream_line_chars_override=int(
            getattr(self.cfg, "repl_bash_max_stream_line_chars", 2400) or 2400
        ),
        bash_observation_summary_override=True,
        interaction_log_layout="split",
        extra_env_override=agent_extra_env,
        skill_registry=self.skill_registry,
        task_type=self.skill_task_category or None,
        skill_allow_names=self.skill_allow_names,
        skill_tool_mode=self.skill_tool_mode,
        skill_allow_generic_wildcard=self.skill_allow_generic_wildcard,
        skill_visible_max=self.skill_visible_max,
        llm_stage_override=llm_stage_override,
    )
    return agent


def _configure_agent_workspace_surface(self, agent: Any) -> None:
    self._resource_main_agent_ref = agent
    # Stage restore/reset intentionally replaces workspace/.logs.  Runtime audit
    # logs must live in the worker control plane so every completed provider call
    # remains traceable across session rollover and stage restore.
    setattr(
        agent,
        "_scienceflow_runtime_log_dir",
        self.log_dir / "agent_runtime_audit",
    )
    next_stage = getattr(self, "_next_stage_id_for_logging", None)
    lineage = getattr(self, "_lineage_uid_prefix", None)
    node = getattr(self, "_stage_node_uid", None)
    active_stage_id = str(next_stage() if callable(next_stage) else "draft") or "draft"
    lineage_id = str(lineage() if callable(lineage) else "")
    setattr(agent, "_scienceflow_stage_id", active_stage_id)
    setattr(agent, "_scienceflow_lineage_id", lineage_id)
    setattr(
        agent,
        "_scienceflow_node_uid",
        str(node(active_stage_id, lineage_id=lineage_id) if callable(node) else ""),
    )
    setattr(
        agent,
        "_scienceflow_run_id",
        str(getattr(self.cfg, "exp_id", "") or self.task_root_dir.name),
    )
    setattr(agent, "_workspace_relative_path_mode", True)
    setattr(agent, "_agent_hidden_workspace_filenames", (self.ledger_filename,))
    setattr(
        agent,
        "_agent_hidden_workspace_path_prefixes",
        (
            self.ledger_filename,
            "logs",
            "submission_snapshots",
            ".logs",
            ".agent_memory",
            ".git",
            ".memory",
            ".scienceflow_checkpoints",
        ),
    )
    hidden_denied_prefixes = (
        "logs",
        ".logs",
        ".agent_memory",
        ".git",
        ".memory",
        ".scienceflow_checkpoints",
        "stage_memory",
        "submission_snapshots",
        "submission_history",
        "submissions",
    )
    apply_denials = getattr(agent, "_apply_agent_hidden_path_denials", None)
    if callable(apply_denials):
        apply_denials(hidden_denied_prefixes)
    setattr(agent, "_agentic_route_log_dir_override", self.log_dir / "agentic_route")
    self._attach_lnr_interaction_logger(agent)


def _configure_agent_bash(self, agent: Any) -> None:
    bash_tool = (
        getattr(getattr(agent, "availableTools", None), "tool_map", {}) or {}
    ).get("bash")
    if bash_tool is not None:
        try:
            bash_tool.forbid_host_absolute_paths = True
            bash_tool.resource_observer = self.resource_observer
            bash_tool.resource_progress_heartbeat_min_interval_sec = float(
                getattr(self.lhr, "resource_progress_heartbeat_min_interval_sec", 30.0)
                or 0.0,
            )
            bash_tool.resource_artifact_heartbeat_scan_interval_sec = float(
                getattr(self.lhr, "resource_artifact_heartbeat_scan_interval_sec", 30.0)
                or 30.0,
            )
            bash_tool.resource_artifact_recoverable_settle_sec = float(
                getattr(self.lhr, "resource_artifact_recoverable_settle_sec", 5.0)
                or 0.0,
            )
            bash_tool.resource_recoverable_stop_enabled = bool(
                getattr(self.lhr, "resource_recoverable_stop_enabled", True),
            )
            bash_tool.resource_recoverable_stop_sigusr1_grace_sec = float(
                getattr(self.lhr, "resource_recoverable_stop_sigusr1_grace_sec", 60.0)
                or 0.0,
            )
            bash_tool.resource_recoverable_stop_marker_exit_grace_sec = float(
                getattr(
                    self.lhr, "resource_recoverable_stop_marker_exit_grace_sec", 10.0
                )
                or 0.0,
            )
            bash_tool.resource_recoverable_stop_sigterm_grace_sec = float(
                getattr(self.lhr, "resource_recoverable_stop_sigterm_grace_sec", 5.0)
                or 0.0,
            )
            finalization_reserve = max(
                0.0,
                float(
                    getattr(
                        self.lhr,
                        "resource_bash_hard_fuse_finalization_reserve_sec",
                        900.0,
                    )
                    or 0.0,
                ),
            )
            remaining_fuse = max(1.0, float(self.deadline - time.monotonic()))
            bash_tool.bash_hard_fuse_deadline_monotonic = float(self.deadline)
            bash_tool.bash_hard_fuse_finalization_reserve_sec = finalization_reserve
            bash_tool.bash_timeout_sec = _effective_lnr_bash_timeout_sec(
                float(getattr(bash_tool, "bash_timeout_sec", 1.0) or 1.0),
                remaining_fuse,
            )
            bash_tool.bash_timeout_slow_sec = _effective_lnr_bash_timeout_sec(
                float(getattr(bash_tool, "bash_timeout_slow_sec", 1.0) or 1.0),
                remaining_fuse,
            )
            setattr(agent, "_bash_timeout_sec", float(bash_tool.bash_timeout_sec))
            setattr(
                agent, "_bash_timeout_slow_sec", float(bash_tool.bash_timeout_slow_sec)
            )
        except Exception:
            logger.debug("[lnr] could not configure bash guards", exc_info=True)
    try:
        core = str(getattr(agent, "_system_prompt_core", "") or "")
        if core:
            setattr(
                agent, "_system_prompt_core", _append_lnr_main_agent_protocols(core)
            )
        else:
            base = str(getattr(agent, "systemPrompt", "") or "")
            setattr(agent, "systemPrompt", _append_lnr_main_agent_protocols(base))
    except Exception:
        logger.debug("[lnr] could not attach LNR main-agent protocols", exc_info=True)


def _configure_agent_protocol(self, agent: Any) -> None:
    self._sanitize_agent_prompt_surfaces(agent)
    if bool(getattr(self.lhr, "clean_repl_mode", True)):
        # LHR is a continuous REPL loop. Keep legacy agent coaching out of
        # the main transcript so text-only replies remain a natural stop/estra signal.
        setattr(agent, "_lnr_phase_header", "")
        setattr(agent, "_guard_manager", None)
        setattr(agent, "_lhr_clean_repl_mode", True)
        setattr(agent, "_lnr_fresh_hint_injected", True)
        setattr(agent, "_lnr_stage_commit_enabled", False)
        setattr(agent, "_lnr_force_stage_journal_after_run", False)
        setattr(agent, "_lnr_stage_journal_pending", False)
    setattr(agent, "_submission_history_archive_enabled", False)
    store = getattr(agent, "_tool_output_artifacts", None)
    setter = getattr(store, "set_mirror_raw_id_prefix", None)
    if callable(setter):
        setter(self._next_stage_id_for_logging())
    setattr(agent, "_lnr_allow_any_stage_script", True)
    install_callback_ports(
        agent,
        callback_ports_for(agent).with_overrides(
            candidate_archive=self._archive_candidate_artifact_after_tool,
            stage_capture=self._stage_capture_callback,
            metric_interpretation=self._metric_output_interpretation_callback,
            text_only_decision=self._text_only_estra_callback,
            context_limit_estra=self._context_limit_estra_callback,
            context_compact_observer=self._record_context_compact_event,
        ),
    )


def _configure_agent_callbacks(self, agent: Any) -> None:
    setattr(
        agent,
        "_lnr_stage_capture_on_candidate_artifact",
        self._evaluator_stage_source_mode() == "primary",
    )
    task_profile = self._evaluator_task_profile()
    evaluator_backend = self._evaluator_backend_name()
    candidate_artifact = self._evaluator_candidate_artifact()
    setattr(agent, "_scienceflow_task_profile", task_profile)
    setattr(agent, "_scienceflow_evaluator_backend", evaluator_backend)
    setattr(agent, "_scienceflow_candidate_artifact_rel", candidate_artifact)
    setattr(agent, "_lnr_candidate_artifact_rel", candidate_artifact)
    try:
        known_artifact_sha = self._candidate_artifact_sha_from_workspace({})
    except Exception:
        known_artifact_sha = ""
    setattr(agent, "_lnr_candidate_artifact_known_sha", known_artifact_sha)
    setattr(agent, "_lnr_candidate_artifact_pending_sha", "")
    setattr(agent, "_scienceflow_evaluation_service", self.evaluation_service)
    setattr(
        agent,
        "_scienceflow_assessment_pipeline",
        getattr(self, "assessment_pipeline", self.evaluation_service),
    )
    setattr(agent, "_scienceflow_evaluation_cfg", self.cfg)
    setattr(agent, "_scienceflow_task_root", self.task_root_dir)
    setattr(agent, "_scienceflow_worker_id", self._worker_uid_prefix())
    self._restore_protected_eda_prefix_marker(agent)
    setattr(
        agent,
        "_lnr_compact_on_context_threshold",
        bool(self.lhr.compact_on_context_limit),
    )
    if bool(getattr(self.lhr, "expose_runtime_context_each_round", False)):
        setattr(
            agent,
            "_lnr_runtime_context_provider",
            lambda: self._runtime_context_for_agent(agent),
        )
    setattr(
        agent,
        "_mlebench_data_dir",
        str(getattr(self.cfg, "mlebench_data_root_dir", "") or "") or None,
    )
    setattr(
        agent, "_mlebench_exp_id", str(getattr(self.cfg, "exp_id", "") or "") or None
    )
    setattr(
        agent,
        "_lnr_mlebench_validate_enabled",
        legacy_score_contract_enabled(
            task_profile=task_profile,
            evaluator_backend=evaluator_backend,
            candidate_artifact=candidate_artifact,
        ),
    )


def _runtime_context_for_agent(self, agent: Any) -> str:
    if _deterministic_gate_enabled():
        remaining = max(
            0.0,
            float(getattr(self.lhr, "wall_clock_budget_sec", 0.0) or 0.0),
        )
    else:
        remaining = max(0.0, float(self.deadline - time.monotonic()))
    configured_bash_timeout = max(
        1.0,
        float(getattr(agent, "_bash_timeout_sec", 1.0) or 1.0),
    )
    effective_bash_timeout = (
        min(configured_bash_timeout, remaining) if remaining > 0 else 0.0
    )
    return (
        "Runtime context (current worker limits for planning the next action):\n"
        f"wall_clock_remaining_sec: {int(remaining)}\n"
        f"effective_bash_timeout_sec: {int(effective_bash_timeout)}"
    )


def _worker_llm_stage_override(self, stage_name: str = "code") -> Any | None:
    """Spread LHR workers across one LLM stage's endpoint pool.

    Regular REPL uses the configured routing unchanged. In LHR multi-worker
    mode, sticky routing can pin each worker to a different primary key.
    When ``agent.<stage>.models`` is configured, it is treated as an indexed
    worker endpoint pool: worker i uses models[i], api_keys[i], base_urls[i]
    with single-value key/url lists broadcast and shorter lists cycled.
    """
    deterministic_gate = _deterministic_gate_enabled()
    worker_count = int(getattr(self, "worker_count", 1) or 1)
    if worker_count <= 1 and not deterministic_gate:
        return None
    stage_name = str(stage_name or "code").strip() or "code"
    stage = getattr(getattr(self.cfg, "agent", None), stage_name, None)
    if stage is None:
        return None

    def _clean_list(value: Any) -> list[str]:
        return [str(x).strip() for x in (value or []) if str(x).strip()]

    models = _clean_list(getattr(stage, "models", []))
    api_keys = _clean_list(getattr(stage, "api_keys", []))
    base_urls = [x.rstrip("/") for x in _clean_list(getattr(stage, "base_urls", []))]
    worker_index = int(getattr(self, "worker_index", 0) or 0)
    seed_worker_index = worker_index
    if str(getattr(stage, "model_selection", "auto") or "auto").lower() == "fixed":
        worker_index = 0
    worker_tag = getattr(self, "worker_id", "") or self._worker_uid_prefix()
    stage_copy = copy.deepcopy(stage)
    aliases = [
        str(item).strip()
        for item in (getattr(stage_copy, "model_aliases", None) or [])
        if str(item).strip()
    ]
    if aliases:
        if (
            str(getattr(stage_copy, "model_selection", "auto") or "auto").lower()
            != "fixed"
        ):
            offset = worker_index % len(aliases)
            aliases = aliases[offset:] + aliases[:offset]
        stage_copy.model_aliases = aliases
        from scienceflow.foundation.config.llm.model_registry import (
            apply_registry_to_stage,
        )

        apply_registry_to_stage(stage_copy, role=stage_name)
        models = _clean_list(getattr(stage_copy, "models", []))
        api_keys = _clean_list(getattr(stage_copy, "api_keys", []))
        base_urls = [
            x.rstrip("/") for x in _clean_list(getattr(stage_copy, "base_urls", []))
        ]
        worker_index = 0
    if deterministic_gate:
        stage_copy.request_seed = max(
            0,
            int(getattr(self.lhr, "seed", 0) or 0) + seed_worker_index,
        )

    if models:
        return _configured_model_stage_override(
            self,
            stage_name=stage_name,
            stage=stage,
            stage_copy=stage_copy,
            models=models,
            api_keys=api_keys,
            base_urls=base_urls,
            worker_index=worker_index,
            worker_tag=worker_tag,
        )

    if len(api_keys) <= 1:
        return stage_copy if deterministic_gate else None
    routing_mode = str(getattr(stage, "api_routing_mode", "") or "").strip().lower()
    if routing_mode not in {
        "sticky",
        "task_sticky",
        "sticky_failover",
        "task_sticky_failover",
    }:
        return stage_copy if deterministic_gate else None
    primary_index = worker_index % len(api_keys)
    base_sticky_id = str(
        getattr(stage_copy, "api_sticky_id", "")
        or getattr(self.cfg, "exp_id", "")
        or "lhr"
    )
    stage_copy.api_sticky_id = f"{base_sticky_id}:{worker_tag}"
    stage_copy.api_sticky_primary_index = primary_index
    self._jsonl(
        "lhr_llm_routing_events.jsonl",
        {
            "event": "worker_sticky_primary_assigned",
            "stage_role": stage_name,
            "worker_id": worker_tag,
            "worker_index": self.worker_index,
            "worker_count": self.worker_count,
            "routing_mode": routing_mode,
            "pool_size": len(api_keys),
            "sticky_primary_index": primary_index,
        },
    )
    return stage_copy


def _configured_model_stage_override(
    self,
    *,
    stage_name: str,
    stage: Any,
    stage_copy: Any,
    models: list[str],
    api_keys: list[str],
    base_urls: list[str],
    worker_index: int,
    worker_tag: str,
) -> Any:
    scalar_key = str(getattr(stage, "api_key", "") or "").strip()
    if not api_keys and scalar_key:
        api_keys = [scalar_key]
    scalar_url = str(getattr(stage, "base_url", "") or "").strip().rstrip("/")
    if not base_urls and scalar_url:
        base_urls = [scalar_url]

    periods = [len(models)]
    if api_keys:
        periods.append(len(api_keys))
    if base_urls and len(base_urls) > 1:
        periods.append(len(base_urls))
    pool_size = max(1, math.lcm(*periods))
    aligned_models = [models[i % len(models)] for i in range(pool_size)]
    aligned_keys = (
        [api_keys[i % len(api_keys)] for i in range(pool_size)] if api_keys else []
    )
    if base_urls:
        aligned_urls = [
            base_urls[0] if len(base_urls) == 1 else base_urls[i % len(base_urls)]
            for i in range(pool_size)
        ]
    else:
        aligned_urls = []

    primary_index = worker_index % pool_size

    def _rotate(values: list[str]) -> list[str]:
        return values[primary_index:] + values[:primary_index]

    rotated_models = _rotate(aligned_models)
    rotated_keys = _rotate(aligned_keys) if aligned_keys else []
    rotated_urls = _rotate(aligned_urls) if aligned_urls else []
    stage_copy.model = rotated_models[0]
    stage_copy.models = rotated_models
    selected_key_index: int | None = None
    selected_url_index: int | None = None
    if rotated_keys:
        selected_key_index = primary_index % len(api_keys)
        stage_copy.api_key = rotated_keys[0]
        stage_copy.api_keys = rotated_keys
    if rotated_urls:
        selected_url_index = (
            0 if len(base_urls) == 1 else primary_index % len(base_urls)
        )
        stage_copy.base_url = rotated_urls[0]
        stage_copy.base_urls = rotated_urls
    if rotated_keys:
        routing_mode = str(getattr(stage, "api_routing_mode", "") or "").strip().lower()
        if routing_mode not in {
            "sticky",
            "task_sticky",
            "sticky_failover",
            "task_sticky_failover",
        }:
            stage_copy.api_routing_mode = "sticky_failover"
        base_sticky_id = str(
            getattr(stage_copy, "api_sticky_id", "")
            or getattr(self.cfg, "exp_id", "")
            or "lhr"
        )
        stage_copy.api_sticky_id = f"{base_sticky_id}:{worker_tag}"
        stage_copy.api_sticky_primary_index = 0
    else:
        stage_copy.api_sticky_id = ""
        stage_copy.api_sticky_primary_index = None
    self._jsonl(
        "lhr_llm_routing_events.jsonl",
        {
            "event": "worker_model_endpoint_assigned",
            "stage_role": stage_name,
            "worker_id": worker_tag,
            "worker_index": self.worker_index,
            "worker_count": self.worker_count,
            "model_index": primary_index % len(models),
            "model_pool_size": len(models),
            "api_key_index": selected_key_index,
            "api_key_pool_size": len(api_keys),
            "base_url_index": selected_url_index,
            "base_url_pool_size": len(base_urls),
            "failover_pool_size": pool_size if rotated_keys else 0,
            "sticky_primary_index": stage_copy.api_sticky_primary_index,
            "pinned_primary_endpoint": bool(rotated_keys or rotated_urls),
        },
    )
    return stage_copy


def _restore_keep_pending_estra(
    self,
    *,
    target: str,
    action: str,
    snap: Any,
    target_node_uid: str,
    summary: str,
    state_packet: str,
    strict_context_limit: bool,
    previous_lineage: str,
) -> None:
    self._prune_workspace_control_artifacts()
    estra_fields = (
        self.pending_estra.get("estra_fields")
        if isinstance(self.pending_estra.get("estra_fields"), dict)
        else {}
    )
    memory_compacted, memory_records = self._rebuild_memory_after_keep_current(
        terminal_stage=target,
        summary=summary,
        state_packet=state_packet,
        strict_context_limit=strict_context_limit,
        continuation_action=action,
        estra_fields=estra_fields,
    )
    self._append_traj_summary(target_stage=target, summary=summary)
    self._reset_interaction_stage_files()
    new_lineage = self._start_new_lineage()
    self.current_restored_from_stage = target
    self.current_restored_from_node_uid = target_node_uid
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_keep_current_compacted",
            "action": action,
            "target_stage": target,
            "target_node_uid": target_node_uid,
            "previous_lineage_id": previous_lineage,
            "new_lineage_id": new_lineage,
            "snapshot_id": snap.snapshot_id,
            "snapshot_path": str(snap.snapshot_path),
            "tail_summary_chars": len(summary),
            "state_packet_chars": len(state_packet),
            "memory_compacted": memory_compacted,
            "memory_records": memory_records,
            "compact_strength": "strict_context_limit"
            if strict_context_limit
            else "normal",
            "context_generation": str(
                self.pending_estra.get("context_generation") or ""
            ),
            "restore_key": str(self.pending_estra.get("restore_key") or ""),
            "preserved_logs": self.log_dir.is_dir(),
            "preserved_memory": (self.memory_dir / "ScienceAgent").is_dir(),
        },
    )
    self.pending_estra = None
    return


async def _restore_pending_estra(self) -> None:
    if not self.pending_estra:
        return
    target = str(self.pending_estra.get("target_stage") or "").upper()
    action = str(self.pending_estra.get("action") or "switch_stage")
    requested_node_uid = str(self.pending_estra.get("target_node_uid") or "").strip()
    snap = self._stage_snapshot_for_restore(
        stage_id=target, node_uid=requested_node_uid
    )
    if snap is None:
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_failed",
                "target_stage": target,
                "target_node_uid": requested_node_uid,
                "reason": "missing_node_snapshot"
                if requested_node_uid
                else "missing_snapshot",
            },
        )
        self.pending_estra = None
        return
    self._close_lnr_interaction_loggers()
    summary = str(self.pending_estra.get("tail_summary") or "")
    state_packet = str(self.pending_estra.get("state_packet") or "")
    target_node_uid = requested_node_uid or self._snapshot_node_uid(snap)
    strict_context_limit = (
        str(self.pending_estra.get("compact_strength") or "") == "strict_context_limit"
    )
    previous_lineage = self._lineage_uid_prefix()
    if self._is_keep_like_estra_action(action):
        _restore_keep_pending_estra(
            self,
            target=target,
            action=action,
            snap=snap,
            target_node_uid=target_node_uid,
            summary=summary,
            state_packet=state_packet,
            strict_context_limit=strict_context_limit,
            previous_lineage=previous_lineage,
        )
        return

    previous_stage_snapshots = dict(self.stage_snapshots)
    previous_restored_stage = str(
        getattr(self, "current_restored_from_stage", "") or ""
    )
    previous_restored_node_uid = str(
        getattr(self, "current_restored_from_node_uid", "") or ""
    )
    archive: Path | None = None
    try:
        archive = self.snapshot_store.restore(snap)
        # The visible Stage ID may also exist in the lineage being abandoned.
        # Keep the active view aligned with the exact archived node restored
        # above; the full node-keyed archive remains available independently.
        self.stage_snapshots[target] = snap
        self._prepare_dataset_symlink()
        self._prune_workspace_control_artifacts()
        append_archived_trajectory_summary(
            self.workspace_dir / self.ledger_filename,
            target_stage=target,
            summary=summary,
        )
        self._prune_active_stage_snapshots_to_ledger()
        self._write_stage_map()
        memory_compacted, memory_records = self._rebuild_memory_after_estra(
            target_stage=target,
            summary=summary,
            state_packet=state_packet,
            strict_context_limit=strict_context_limit,
        )
        self._append_traj_summary(target_stage=target, summary=summary)
        self._reset_interaction_stage_files()
    except Exception as exc:
        rollback_attempted = archive is not None
        rollback_succeeded = archive is None
        rollback_error = ""
        if archive is not None:
            try:
                self.snapshot_store.restore_terminal_archive(archive)
                rollback_succeeded = True
            except Exception as rollback_exc:
                rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
                logger.exception("[lnr] estra terminal workspace rollback failed")
        self.stage_snapshots = previous_stage_snapshots
        self.current_restored_from_stage = previous_restored_stage
        self.current_restored_from_node_uid = previous_restored_node_uid
        try:
            self._write_stage_map()
        except Exception:
            logger.debug("[lnr] stage map rollback write failed", exc_info=True)
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_failed",
                "action": action,
                "target_stage": target,
                "target_node_uid": target_node_uid,
                "reason": "unfold_transaction_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "rollback_attempted": rollback_attempted,
                "rollback_succeeded": rollback_succeeded,
                "rollback_error": rollback_error,
                "lineage_id": previous_lineage,
            },
        )
        self.pending_estra = None
        return

    new_lineage = self._start_new_lineage()
    self.current_restored_from_stage = target
    self.current_restored_from_node_uid = target_node_uid
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_stage_switched",
            "target_stage": target,
            "target_node_uid": target_node_uid,
            "previous_lineage_id": previous_lineage,
            "new_lineage_id": new_lineage,
            "snapshot_id": snap.snapshot_id,
            "snapshot_path": str(snap.snapshot_path),
            "terminal_archive": str(archive),
            "tail_summary_chars": len(summary),
            "state_packet_chars": len(state_packet),
            "memory_compacted": memory_compacted,
            "memory_records": memory_records,
            "compact_strength": "strict_context_limit"
            if strict_context_limit
            else "normal",
            "context_generation": str(
                self.pending_estra.get("context_generation") or ""
            ),
            "restore_key": str(self.pending_estra.get("restore_key") or ""),
            "preserved_logs": self.log_dir.is_dir(),
            "preserved_memory": (self.memory_dir / "ScienceAgent").is_dir(),
        },
    )
    self.pending_estra = None


def _accumulate_main_run_tokens(self, agent: Any) -> None:
    self.main_tokens_in += int(getattr(agent, "last_run_tokens_in", 0) or 0)
    self.main_tokens_out += int(getattr(agent, "last_run_tokens_out", 0) or 0)
    self.main_tokens_cached += int(getattr(agent, "last_run_tokens_cached", 0) or 0)
    self.main_llm_calls += int(getattr(agent, "last_run_llm_calls", 0) or 0)


def _result(self, *, stop_reason: str) -> dict[str, Any]:
    main_cache_rate = (
        self.main_tokens_cached / self.main_tokens_in if self.main_tokens_in else 0.0
    )
    estra_counts = self._estra_decision_counts()
    best_stage = None
    best_metric = None
    lower = True
    for sid, snap in self.stage_snapshots.items():
        if snap.metric_value is None:
            continue
        if best_metric is None:
            best_metric = snap.metric_value
            best_stage = sid
            lower = bool(snap.lower_is_better is not False)
            continue
        if lower and snap.metric_value < best_metric:
            best_metric = snap.metric_value
            best_stage = sid
        elif not lower and snap.metric_value > best_metric:
            best_metric = snap.metric_value
            best_stage = sid
    stage_count = len(self.stage_snapshots)
    outcome = "finalize_candidate" if stage_count else "stop_no_candidate"
    return {
        "solver": self.solver_name,
        "status": "success" if stage_count else "no_candidate",
        "stop_reason": stop_reason,
        "outcome": outcome,
        "failure_kind": "" if stage_count else "no_stage_candidate",
        "stage_count": stage_count,
        "estra_switch_stage_count": self._count_jsonl_events(
            "lhr_estras.jsonl", "estra_stage_switched"
        ),
        "estra_compact_count": self._count_jsonl_events(
            "lhr_estras.jsonl", "estra_keep_current_compacted"
        ),
        "estra_decisions": estra_counts.get("decisions", self.estra_decisions),
        "estra_fallback_count": estra_counts.get("fallback", 0),
        "estra_continue_count": estra_counts["continue"],
        "estra_redirect_count": estra_counts["redirect"],
        "estra_switch_count": estra_counts["switch"],
        "estra_current_continue_count": estra_counts["current_continue"],
        "estra_current_redirect_count": estra_counts["current_redirect"],
        "estra_stage_continue_count": estra_counts["stage_continue"],
        "estra_stage_redirect_count": estra_counts["stage_redirect"],
        "best_stage": best_stage,
        "best_metric": best_metric,
        "score_summary": self._score_summary_for_prompt(),
        "main_cache_rate": main_cache_rate,
        "main_tokens_input": self.main_tokens_in,
        "main_tokens_cached": self.main_tokens_cached,
        "main_tokens_output": self.main_tokens_out,
        "main_llm_calls": self.main_llm_calls,
        "stage_tokens_input": self.stage_tokens_in,
        "stage_tokens_cached": self.stage_tokens_cached,
        "stage_llm_calls": self.stage_llm_calls,
        "estra_tokens_input": self.estra_tokens_in,
        "estra_tokens_cached": self.estra_tokens_cached,
        "estra_llm_calls": self.estra_llm_calls,
        "workspace_dir": str(self.workspace_dir),
        "ledger": str(self.ledger_path),
    }
