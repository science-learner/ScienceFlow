# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: results, workers, merge, finalization, and public run dispatch.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    LHR_EVENTS_JSONL,
    LHR_UNIFIED_ONLY_JSONL,
    Any,
    Config,
    LHRStateMachineStore,
    LnrRuntime,
    Path,
    _WallClockAutoContinuePolicy,
    aclose_llm_clients,
    apply_candidate_evidence,
    asyncio,
    build_worker_environment,
    copy,
    csv,
    ensure_lnr_worker_layout,
    format_cpu_ids,
    global_merge_reserve_sec,
    json,
    lnr_worker_layout,
    load_archived_artifact_candidates,
    load_peer_candidate_evidence,
    logger,
    metric_float,
    multi_worker_failure_kind,
    multi_worker_stop_reason,
    recover_candidate_artifact,
    refresh_submission_links,
    run_multi_worker,
    run_single_worker,
    shutil,
    slice_cpu_ids,
    time,
    worker_error_kind,
    worker_wall_clock_budget_sec,
)


def _count_jsonl_events(self, name: str, event: str) -> int:
    paths = [self.log_dir / name]
    if name in LHR_UNIFIED_ONLY_JSONL or name == LHR_EVENTS_JSONL:
        paths = [self.log_dir / LHR_EVENTS_JSONL]
    n = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("event") == event:
                    n += 1
        except (OSError, json.JSONDecodeError):
            return n
    return n


def _estra_decision_counts(self) -> dict[str, int]:
    counts = {
        "decisions": 0,
        "continue": 0,
        "redirect": 0,
        "switch": 0,
        "current_continue": 0,
        "current_redirect": 0,
        "stage_continue": 0,
        "stage_redirect": 0,
        "fallback": 0,
    }
    path = self.log_dir / LHR_EVENTS_JSONL
    if not path.is_file():
        return counts
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return counts
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        event = str(obj.get("event") or "")
        if event not in {"estra_decision", "estra_deterministic_fallback"}:
            continue
        legacy_fallback = event == "estra_deterministic_fallback"
        action = str(
            payload.get("action")
            or payload.get("estra_action")
            or obj.get("action")
            or obj.get("estra_action")
            or "keep_current"
        )
        startpoint = str(
            payload.get("startpoint")
            or obj.get("startpoint")
            or ("previous_stage" if action == "switch_stage" else "current_workspace")
        )
        intent = str(
            payload.get("intent")
            or obj.get("intent")
            or ("redirect" if action == "keep_but_redirect" else "continue")
        )
        kind = self._estra_decision_kind(
            action=action, startpoint=startpoint, intent=intent
        )
        decision_mode = (
            "deterministic_fallback"
            if legacy_fallback
            else str(payload.get("decision_mode") or obj.get("decision_mode") or "")
        )
        counts["decisions"] += 1
        if decision_mode in {"safe_fallback", "deterministic_fallback"}:
            counts["fallback"] += 1
        if kind in {"continue", "redirect", "switch"}:
            counts[kind] += 1
        if action in {"keep_current", "keep_but_redirect", "switch_stage"}:
            if startpoint == "previous_stage" and intent == "redirect":
                counts["stage_redirect"] += 1
            elif startpoint == "previous_stage":
                counts["stage_continue"] += 1
            elif intent == "redirect":
                counts["current_redirect"] += 1
            else:
                counts["current_continue"] += 1
    return counts


@staticmethod
def _format_cpu_ids(cpu_ids: list[int]) -> str:
    return format_cpu_ids(cpu_ids)


@staticmethod
def _slice_cpu_ids(
    cpu_ids: list[int], *, worker_index: int, worker_count: int
) -> list[int]:
    return slice_cpu_ids(
        cpu_ids,
        worker_index=worker_index,
        worker_count=worker_count,
    )


def _worker_extra_env(self, *, worker_index: int, worker_count: int) -> dict[str, str]:
    return build_worker_environment(
        cfg=self.cfg,
        lhr=self.lhr,
        worker_index=worker_index,
        worker_count=worker_count,
    )


def _worker_root(self, worker_index: int) -> Path:
    return lnr_worker_layout(self.root_dir, self.lhr, worker_index).root


def _worker_log_dir(self, worker_index: int) -> Path:
    return lnr_worker_layout(self.root_dir, self.lhr, worker_index).logs


def _merge_dir(self) -> Path:
    dirname = (
        str(getattr(self.lhr, "merge_dirname", "merge") or "merge").strip() or "merge"
    )
    return self.root_dir / dirname


def _global_merge_reserve_sec(self) -> float:
    return global_merge_reserve_sec(self.lhr)


def _final_artifact_mode(self) -> str:
    mode = (
        str(getattr(self.lhr, "final_artifact_mode", "workspace") or "workspace")
        .strip()
        .lower()
    )
    if mode not in {"workspace", "best_stage"}:
        raise ValueError(f"unsupported lnr.final_artifact_mode: {mode!r}")
    return mode


def _worker_wall_clock_budget_sec(self) -> int:
    return worker_wall_clock_budget_sec(self.lhr)


def _refresh_submission_links(self, *, n_workers: int, include_merge: bool) -> None:
    raw_submission_dir = getattr(self.cfg, "submission_dir", None)
    if raw_submission_dir is None or not str(raw_submission_dir).strip():
        return
    try:
        links = refresh_submission_links(
            submission_dir=Path(raw_submission_dir),
            artifact_path=self._evaluator_candidate_artifact(),
            merge_dir=self._merge_dir() if include_merge else None,
            worker_roots=[
                self._worker_root(i) for i in range(max(1, int(n_workers or 1)))
            ],
        )
        self._jsonl(
            "lhr_coordinator_events.jsonl",
            {
                "event": "submission_links_refreshed",
                "link_count": len(links),
                "include_merge": include_merge,
                "submission_dir": str(raw_submission_dir),
            },
        )
    except Exception as exc:  # noqa: BLE001 - submission links are convenience outputs.
        logger.warning("failed to refresh LNR submission links: %s", exc)


def _cleanup_coordinator_workspace_shell(self) -> None:
    if self.worker_id or max(1, int(self.lhr.num_workers or 1)) <= 1:
        return
    ws = self.workspace_dir
    if not ws.exists() or not ws.is_dir():
        return
    # Older layouts created a minimal coordinator workspace. In worker-indexed
    # mode actual agents live under workers/wXX, so remove the empty shell
    # only when it is clearly not the task root.
    if ws.resolve(strict=False) == self.root_dir.resolve(strict=False):
        return
    allowed = {"logs", "submissions"}
    try:
        names = {p.name for p in ws.iterdir()}
    except OSError:
        return
    if not names.issubset(allowed):
        return
    for child in sorted(ws.iterdir(), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir():
            try:
                shutil.rmtree(child)
            except OSError:
                logger.debug(
                    "[lnr] coordinator workspace cleanup skipped", exc_info=True
                )
                return
    try:
        ws.rmdir()
    except OSError:
        logger.debug("[lnr] coordinator workspace shell not empty", exc_info=True)


def _cleanup_worker_root_artifacts(self, worker_root: Path) -> None:
    try:
        (worker_root / "resolved_config.yaml").unlink(missing_ok=True)
    except OSError:
        logger.debug("[lnr] worker resolved_config cleanup skipped", exc_info=True)


def _aggregate_worker_state(self, *, n_workers: int, run_status: str) -> dict[str, Any]:
    log_dirs = [self.global_log_dir]
    for idx in range(max(1, int(n_workers or 1))):
        log_dirs.append(self._worker_log_dir(idx))
    try:
        return LHRStateMachineStore.aggregate_logs(
            output_log_dir=self.global_log_dir,
            input_log_dirs=log_dirs,
            worker_count=n_workers,
            run_status=run_status,
            ledger_filename=self.ledger_filename,
        )
    except OSError:
        logger.debug("[lnr] worker state aggregation failed", exc_info=True)
        return {}


async def _aggregate_worker_state_periodically(
    self,
    *,
    n_workers: int,
    run_status: str = "running",
    interval_sec: float = 60.0,
) -> None:
    """Refresh coordinator LHR state while worker-indexed runs are active."""

    interval = max(5.0, float(interval_sec or 60.0))
    while True:
        try:
            self._aggregate_worker_state(n_workers=n_workers, run_status=run_status)
        except Exception:  # noqa: BLE001 - monitor state refresh must not affect workers.
            logger.debug("[lnr] live worker state aggregation failed", exc_info=True)
        await asyncio.sleep(interval)


def _write_global_time_trace(self, worker_results: list[dict[str, Any]]) -> None:
    from scienceflow.runtime.observability.telemetry.trace.time_trace import (
        TRACE_COLUMNS,
        TRACE_FILENAME,
    )

    out = self.global_log_dir / TRACE_FILENAME
    rows: list[dict[str, str]] = []
    for result in worker_results:
        worker_id = str(result.get("worker_id") or "")
        worker_index = str(result.get("worker_index") or "")
        try:
            worker_index_int = int(result.get("worker_index") or 0)
        except (TypeError, ValueError):
            continue
        src = self._worker_log_dir(worker_index_int) / TRACE_FILENAME
        if not src.is_file():
            continue
        try:
            with src.open("r", encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.DictReader(f):
                    merged = {"worker_id": worker_id, "worker_index": worker_index}
                    for col in TRACE_COLUMNS:
                        merged[col] = str(row.get(col) or "")
                    rows.append(merged)
        except OSError:
            logger.debug("[lnr] worker time trace read failed: %s", src, exc_info=True)
    if not rows:
        return
    try:
        self.global_log_dir.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["worker_id", "worker_index", *TRACE_COLUMNS]
            )
            writer.writeheader()
            writer.writerows(rows)
        for result in worker_results:
            try:
                (
                    self._worker_log_dir(int(result.get("worker_index") or 0))
                    / TRACE_FILENAME
                ).unlink(missing_ok=True)
            except (OSError, TypeError, ValueError):
                pass
    except OSError:
        logger.debug("[lnr] global time trace write failed", exc_info=True)


@staticmethod
def _worker_error_kind(error: str) -> str:
    return worker_error_kind(error)


@classmethod
def _multi_worker_failure_kind(
    cls, worker_results: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    return multi_worker_failure_kind(worker_results)


@classmethod
def _multi_worker_stop_reason(
    cls, *, run_succeeded: bool, worker_results: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    if run_succeeded:
        return multi_worker_stop_reason(
            run_succeeded=True,
            worker_results=worker_results,
        )
    # Preserve the legacy empty stop reason on failed coordinator runs.  The
    # structured failure kind is emitted separately.
    return "", multi_worker_failure_kind(worker_results)[1]


def _make_worker_cfg(self, worker_index: int, worker_count: int) -> Config:
    cfg = copy.deepcopy(self.cfg)
    layout = ensure_lnr_worker_layout(self.root_dir, self.lhr, worker_index)
    cfg.task_workspace_root_dir = layout.root
    setattr(cfg, "_workspace_dir_override", layout.workspace)
    setattr(cfg, "_log_dir_override", layout.logs)
    cfg.lnr.num_workers = 1
    cfg.lnr.merge_enabled = False
    cfg.lnr.wall_clock_budget_sec = self._worker_wall_clock_budget_sec()
    return cfg


async def _run_one_worker(self, worker_index: int, worker_count: int) -> dict[str, Any]:
    worker_id = f"W{worker_index:02d}"
    from scienceflow.research.solver.lnr.orchestration.coordinator.solver import (
        LnrSolver,
    )
    from scienceflow.runtime.workflow import Orchestrator

    worker_cfg = self._make_worker_cfg(worker_index, worker_count)
    worker_env = self._worker_extra_env(
        worker_index=worker_index,
        worker_count=worker_count,
    )
    self._jsonl(
        "lhr_coordinator_events.jsonl",
        {
            "event": "worker_start",
            "worker_id": worker_id,
            "worker_index": worker_index,
            "worker_root": str(
                Path(worker_cfg.task_workspace_root_dir).relative_to(self.root_dir)
            ),
            "cpu_set": worker_env.get("_SCIENCEFLOW_CPU_SET", ""),
            "omp_threads": worker_env.get("OMP_NUM_THREADS", ""),
            "worker_wall_clock_budget_sec": int(
                worker_cfg.lnr.wall_clock_budget_sec or 0
            ),
            "global_merge_reserve_sec": int(self._global_merge_reserve_sec()),
        },
    )
    worker_orchestrator = Orchestrator(worker_cfg)
    worker_solver = LnrSolver(
        task_desc=self.task_desc,
        cfg=worker_orchestrator.cfg,
        orchestrator=worker_orchestrator,
        worker_id=worker_id,
        worker_index=worker_index,
        worker_count=worker_count,
        worker_extra_env=worker_env,
        task_root_dir=self.root_dir,
    )
    merge_owner = str(getattr(self.lhr, "merge_owner_worker", "W00") or "W00")
    keep_agent_open = (
        bool(getattr(self.lhr, "merge_enabled", True))
        and str(getattr(self.lhr, "merge_mode", "worker_reduce") or "worker_reduce")
        == "worker_reduce"
        and worker_id == merge_owner
    )
    try:
        result = await worker_solver._run_single(keep_agent_open=keep_agent_open)
        if keep_agent_open and worker_solver._live_agent is not None:
            self._merge_owner_solver = worker_solver
        return {"worker_id": worker_id, "worker_index": worker_index, **result}
    except BaseException as exc:
        logger.exception("[lnr] worker %s failed", worker_id)
        return {
            "worker_id": worker_id,
            "worker_index": worker_index,
            "solver": self.solver_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "workspace_dir": str(
                getattr(
                    worker_cfg,
                    "_workspace_dir_override",
                    worker_cfg.task_workspace_root_dir,
                )
            ),
        }
    finally:
        self._cleanup_worker_root_artifacts(Path(worker_cfg.task_workspace_root_dir))


@staticmethod
def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _candidate_from_stage(
    self,
    *,
    worker_id: str,
    worker_root: Path,
    stage_id: str,
    stage: dict[str, Any],
) -> dict[str, Any]:
    snapshot_path = Path(str(stage.get("snapshot_path") or ""))
    meta = (
        self._read_json_file(snapshot_path / "logs" / "lhr_snapshot_meta.json")
        if snapshot_path
        else {}
    )
    source = (
        meta.get("source_event") if isinstance(meta.get("source_event"), dict) else {}
    )
    lineage_id = str(stage.get("lineage_id") or source.get("lineage_id") or "")
    node_uid = str(stage.get("node_uid") or source.get("node_uid") or "")
    candidate_id = node_uid or ":".join(
        part for part in (worker_id, lineage_id, stage_id) if part
    )
    return {
        "candidate_id": candidate_id,
        "worker_id": worker_id,
        "stage_id": stage_id,
        "worker_root": str(worker_root),
        "workspace_path": str(worker_root / "workspace"),
        "snapshot_id": str(stage.get("snapshot_id") or ""),
        "snapshot_path": str(snapshot_path) if str(snapshot_path) else "",
        "metric_name": str(
            stage.get("metric_name")
            or source.get("metric_name")
            or "Final Validation Score"
        ),
        "metric_value": stage.get("metric_value"),
        "lower_is_better": stage.get("lower_is_better"),
        "memory_cut": stage.get("memory_cut"),
        "validation_ok": source.get("validation_ok"),
        "validation_issue": source.get("validation_issue")
        or stage.get("validation_issue")
        or "",
        "reported_val_score": source.get("reported_val_score")
        or stage.get("reported_val_score"),
        "val_score_type": source.get("val_score_type")
        or stage.get("val_score_type")
        or "",
        "selection_eligible": (
            source.get("selection_eligible")
            if "selection_eligible" in source
            else stage.get("selection_eligible")
        ),
        "selection_score": source.get("selection_score")
        or stage.get("selection_score")
        or "",
        "selection_note": source.get("selection_note")
        or stage.get("selection_note")
        or "",
        "metric_source_note": source.get("metric_source_note")
        or stage.get("metric_source_note")
        or "",
        "metric_protocol": source.get("metric_protocol")
        or stage.get("metric_protocol")
        or "",
        "train_data_used": source.get("train_data_used")
        or stage.get("train_data_used")
        or "",
        "metric_eval_data": source.get("metric_eval_data")
        or stage.get("metric_eval_data")
        or "",
        "execution_mode": source.get("execution_mode")
        or stage.get("execution_mode")
        or "",
        "metric_validity": source.get("metric_validity")
        or stage.get("metric_validity")
        or "",
        "metric_validity_note": source.get("metric_validity_note")
        or stage.get("metric_validity_note")
        or "",
        "metric_validity_reason_code": source.get("metric_validity_reason_code")
        or stage.get("metric_validity_reason_code")
        or "",
        "metric_authoritative": source.get("metric_authoritative") is True,
        "evaluator_backend": str(source.get("evaluator_backend") or ""),
        "lineage_id": lineage_id,
        "node_uid": node_uid,
        "route_id": str(source.get("route_id") or stage.get("route_id") or ""),
        "solution_sha": str(
            source.get("solution_sha") or stage.get("solution_sha") or ""
        ),
        "submission_sha": str(
            source.get("submission_sha") or stage.get("submission_sha") or ""
        ),
        "submission_snapshot": str(
            source.get("submission_snapshot") or stage.get("submission_snapshot") or ""
        ),
        "artifact_path": str(
            source.get("artifact_path") or stage.get("artifact_path") or ""
        ),
        "artifact_sha": str(
            source.get("artifact_sha") or stage.get("artifact_sha") or ""
        ),
        "candidate_ready": (
            source.get("candidate_ready")
            if "candidate_ready" in source
            else stage.get("candidate_ready")
            if "candidate_ready" in stage
            else bool(
                source.get("artifact_sha")
                or stage.get("artifact_sha")
                or source.get("submission_sha")
                or stage.get("submission_sha")
            )
        ),
        "submission_status": str(source.get("submission_status") or ""),
        "duplicate_submission_of_stage": str(
            source.get("duplicate_submission_of_stage") or ""
        ),
    }


def _load_worker_candidates(
    self,
    worker_result: dict[str, Any],
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    worker_id = str(worker_result.get("worker_id") or "")
    worker_index = int(worker_result.get("worker_index") or 0)
    worker_root = self._worker_root(worker_index)
    stage_map = self._read_json_file(
        self._worker_log_dir(worker_index) / "lhr_stage_map.json"
    )
    stages = (
        stage_map.get("stages") if isinstance(stage_map.get("stages"), dict) else {}
    )
    archive = (
        stage_map.get("archive") if isinstance(stage_map.get("archive"), dict) else {}
    )
    peer_evidence = load_peer_candidate_evidence(
        self.log_dir / "lhr_stage_performance.csv",
        worker_id=worker_id,
    )
    candidates: list[dict[str, Any]] = []
    entries: list[tuple[str, dict[str, Any], bool]] = [
        (str(stage_id), stage, False)
        for stage_id, stage in sorted(stages.items())
        if isinstance(stage, dict)
    ]
    active_nodes = {
        str(stage.get("node_uid") or "").strip()
        for stage in stages.values()
        if isinstance(stage, dict) and str(stage.get("node_uid") or "").strip()
    }
    if include_archived:
        entries.extend(
            (
                str(stage.get("stage_id") or ""),
                stage,
                True,
            )
            for node_uid, stage in sorted(archive.items())
            if isinstance(stage, dict) and str(node_uid).strip() not in active_nodes
        )
    for stage_id, stage, is_archived in entries:
        if isinstance(stage, dict):
            candidate = self._candidate_from_stage(
                worker_id=worker_id,
                worker_root=worker_root,
                stage_id=str(stage_id),
                stage=stage,
            )
            if not is_archived:
                candidate = apply_candidate_evidence(
                    candidate,
                    peer_evidence.get(str(stage_id).upper()),
                )
            candidates.append(
                recover_candidate_artifact(
                    candidate,
                    artifact_path=self._evaluator_candidate_artifact(),
                )
            )
    ready_artifact_shas = {
        str(candidate.get("artifact_sha") or candidate.get("submission_sha") or "")
        .strip()
        .lower()
        for candidate in candidates
        if candidate.get("candidate_ready") is True
    }
    candidates.extend(
        candidate
        for candidate in load_archived_artifact_candidates(
            worker_root,
            worker_id=worker_id,
            artifact_path=self._evaluator_candidate_artifact(),
        )
        if str(candidate.get("artifact_sha") or "").strip().lower()
        not in ready_artifact_shas
    )
    return candidates


@staticmethod
def _metric_float(candidate: dict[str, Any]) -> float | None:
    return metric_float(candidate)


def _write_stage_collection_outputs(
    self,
    *,
    worker_results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> Path:
    collection_dir = self._merge_dir()
    collection_dir.mkdir(parents=True, exist_ok=True)
    (collection_dir / "global_candidates.jsonl").write_text(
        "".join(
            json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in candidates
        ),
        encoding="utf-8",
    )
    (collection_dir / "worker_results.json").write_text(
        json.dumps(worker_results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "candidates": candidates,
        "merge": {
            "merge_mode": "not_run",
            "reason": "global merge has not run",
        },
    }
    (collection_dir / "global_stage_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Long Horizon REPL Stage Collection",
        "",
        f"candidate_count: {len(candidates)}",
        f"worker_count: {len(worker_results)}",
        "global_merge: not_run",
        "",
        "## Workers",
    ]
    for result in worker_results:
        report.append(
            f"- {result.get('worker_id')}: status={result.get('status')} "
            f"best_stage={result.get('best_stage')} best_metric={result.get('best_metric')}"
        )
    report.extend(["", "## Stage Candidates"])
    for candidate in candidates:
        report.append(
            f"- {candidate.get('candidate_id')}: metric={candidate.get('metric_value')} "
            f"snapshot={candidate.get('snapshot_path')} submission_sha={candidate.get('submission_sha')}"
        )
    (collection_dir / "stage_collection_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return collection_dir


async def _run_live_merge_agent(
    self,
    workspace: Path,
    prompt: str,
    budget_sec: float,
    read_roots: list[Path],
) -> None:
    owner = getattr(self, "_merge_owner_solver", None)
    agent = owner._live_agent if owner is not None else None
    if owner is None or agent is None:
        raise RuntimeError("configured merge owner has no live agent")

    reduction_deadline = min(
        self.deadline,
        time.monotonic() + max(30.0, float(budget_sec or 0.0)),
    )
    owner.deadline = reduction_deadline
    hook = owner.orchestrator.make_llm_call_tracer(
        node_id=f"lnr:{owner.worker_id}:global_merge",
        process_id=(
            f"{getattr(owner.cfg, 'exp_id', '') or 'lhr'}:"
            f"{owner.worker_id}:global_merge"
        ),
        detail_prefix=f"mode=lnr;role=global_merge;worker={owner.worker_id}",
    )
    policy = _WallClockAutoContinuePolicy(
        deadline_monotonic=reduction_deadline,
        max_text_only_retries=0,
    )
    agent.swap_workspace(
        workspace_dir=workspace,
        path_guard_extra_roots=read_roots,
        readonly_dirs=["candidates", "dataset"],
        run_policy=policy,
        max_steps_override=40,
        bash_timeout_sec=max(30.0, reduction_deadline - time.monotonic()),
        bash_timeout_slow_sec=max(30.0, reduction_deadline - time.monotonic()),
        extra_env={
            **owner.worker_extra_env,
            **owner._task_runtime_extra_env(),
        },
        skill_registry=owner.skill_registry,
        task_type=owner.skill_task_category or None,
        skill_allow_names=owner.skill_allow_names,
        skill_tool_mode=owner.skill_tool_mode,
        skill_allow_generic_wildcard=owner.skill_allow_generic_wildcard,
        skill_visible_max=owner.skill_visible_max,
        resource_observer=owner.resource_observer,
        interaction_log_session="global_merge",
        interaction_log_phase="reduce",
        copy_memory_storage_to_new_workspace=False,
        on_llm_call=hook,
    )
    bash_tool = (
        getattr(getattr(agent, "availableTools", None), "tool_map", {}) or {}
    ).get("bash")
    if bash_tool is not None:
        bash_tool.forbid_host_absolute_paths = True
        bash_tool.bash_hard_fuse_deadline_monotonic = reduction_deadline
        bash_tool.bash_hard_fuse_finalization_reserve_sec = 0.0

    owner.state_machine.mark_run_status(
        "reducing",
        payload={"worker_id": owner.worker_id, "workspace": str(workspace)},
    )
    request: str | None = prompt
    while time.monotonic() < reduction_deadline:
        await agent.run(request)
        owner._accumulate_main_run_tokens(agent)
        request = None
        artifact = owner._evaluator_candidate_artifact()
        if any(
            (path / artifact).is_file()
            for path in (workspace / "finals").glob("final_*")
            if path.is_dir()
        ):
            break
        await asyncio.sleep(0.1)


async def _close_merge_owner_agent(self) -> None:
    owner = getattr(self, "_merge_owner_solver", None)
    if owner is None or owner._live_agent is None:
        return
    agent = owner._live_agent
    owner._live_agent = None
    self._merge_owner_solver = None
    try:
        owner.state_machine.mark_run_status(
            "finished",
            payload={"worker_id": owner.worker_id, "phase": "global_merge"},
        )
    except OSError:
        logger.debug("[lnr] merge-owner final status write skipped", exc_info=True)
    try:
        await aclose_llm_clients(agent.llm)
    except Exception:
        logger.debug("[lnr] merge-owner llm close skipped", exc_info=True)


async def _write_merge_outputs(
    self,
    *,
    worker_results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    merge_dir = self._merge_dir()
    configured_wall_clock_sec = float(
        getattr(self.lhr, "global_merge_wall_clock_sec", 900.0) or 900.0
    )
    remaining_sec = max(0.0, float(self.deadline - time.monotonic()))
    wall_clock_sec = min(configured_wall_clock_sec, max(0.0, remaining_sec - 60.0))
    artifact_path = self._evaluator_candidate_artifact()
    owner = getattr(self, "_merge_owner_solver", None)
    merge_workspace = (
        owner.workspace_dir / ".scienceflow_global_merge"
        if owner is not None
        else merge_dir / "global_merge_workspace"
    )

    dataset_source = Path(self.cfg.input_data_dir).expanduser().resolve(strict=False)
    manifest = await self.finalization_service.finalize_global(
        merge_dir=merge_dir,
        candidates=candidates,
        worker_results=worker_results,
        task_desc=self.task_desc,
        artifact_path=artifact_path,
        ledger_filename=self.ledger_filename,
        wall_clock_sec=wall_clock_sec,
        evaluator_manager=self.evaluator_manager,
        cfg=self.cfg,
        task_profile=self._evaluator_task_profile(),
        task_id=str(getattr(self.cfg, "exp_id", "") or ""),
        task_root=self.task_root_dir,
        dataset_source=dataset_source,
        merge_executor=self._run_live_merge_agent if owner is not None else None,
        workspace_override=merge_workspace,
        max_prediction_file_bytes=int(
            getattr(self.lhr, "merge_prediction_file_max_bytes", 536_870_912) or 0
        ),
        max_prediction_total_bytes=int(
            getattr(self.lhr, "merge_prediction_total_max_bytes", 2_147_483_648) or 0
        ),
        required_finals=max(
            1,
            int(getattr(self.lhr, "merge_required_finals", 3) or 3),
        ),
        max_finals=max(
            1,
            int(getattr(self.lhr, "merge_max_finals", 3) or 3),
            int(getattr(self.lhr, "merge_required_finals", 3) or 3),
        ),
    )
    payload = {"candidates": candidates, "merge": manifest}
    (merge_dir / "global_stage_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Long Horizon REPL Merge Report",
        "",
        f"status: {manifest.get('status')}",
        f"merge_mode: {manifest.get('merge_mode')}",
        f"global_merge_workspace: {manifest.get('workspace')}",
        "",
        "## Workers",
    ]
    for result in worker_results:
        report.append(
            f"- {result.get('worker_id')}: status={result.get('status')} "
            f"best_stage={result.get('best_stage')} best_metric={result.get('best_metric')} "
            f"main_cache_rate={result.get('main_cache_rate')}"
        )
    report.extend(
        [
            "",
            "## Global Merge",
            f"agent_status: {manifest.get('agent_status')}",
            f"packed_candidate_count: {manifest.get('packed_candidate_count')}",
            f"required_final_count: {manifest.get('required_final_count')}",
            f"final_count: {manifest.get('final_count')}",
            f"valid_final_count: {manifest.get('valid_final_count')}",
            f"requirement_met: {manifest.get('requirement_met')}",
            f"merge_budget_sec: {wall_clock_sec:.1f}",
            "manifest: global_merge_manifest.json",
        ]
    )
    (merge_dir / "merge_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return manifest


async def _run_multi_worker(self) -> dict[str, Any]:
    from scienceflow.research.solver.lnr.orchestration.runtime.execution.adapters import (
        coordinator_services_from_legacy_host,
    )

    return await run_multi_worker(coordinator_services_from_legacy_host(self))


async def _run_single(self, *, keep_agent_open: bool = False) -> dict[str, Any]:
    from scienceflow.research.solver.lnr.orchestration.runtime.execution.adapters import (
        worker_runtime_services_from_legacy_host,
    )

    return await run_single_worker(
        worker_runtime_services_from_legacy_host(self),
        keep_agent_open=keep_agent_open,
    )


async def run(self) -> dict[str, Any]:
    from scienceflow.research.solver.lnr.orchestration.runtime.execution.adapters import (
        runtime_services_from_legacy_host,
    )

    runtime = getattr(self, "runtime", None)
    if not isinstance(runtime, LnrRuntime):
        runtime = LnrRuntime(runtime_services_from_legacy_host(self))
        self.runtime = runtime
    return await runtime.run()


@staticmethod
async def _ask_agent_tool_stream_guarded(
    agent: Any,
    *,
    llm: Any | None = None,
    **kwargs: Any,
) -> Any:
    guarded = getattr(agent, "_ask_tool_stream_guarded", None)
    if callable(guarded):
        return await guarded(llm=llm, **kwargs)
    raise RuntimeError(
        "Legacy direct provider calls are disabled; use the unified agent runtime"
    )
