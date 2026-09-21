# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: configuration, factories, and resource-observer construction.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scienceflow.research.solver.lnr.orchestration.coordinator.solver import LnrSolver

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    AgentFactory,
    Any,
    Config,
    CandidateAssessmentService,
    LHRStateMachineStore,
    Path,
    SkillRegistry,
    SnapshotStore,
    StageSnapshot,
    build_lnr_module_graph,
    default_skill_library_dir,
    ensure_lnr_context_memory_budget,
    json,
    logger,
    metric_lower_is_better_hint,
    os,
    time,
)


def __init__(
    self,
    *,
    task_desc: str,
    cfg: Config,
    orchestrator: Any,
    worker_id: str = "",
    worker_index: int = 0,
    worker_count: int = 1,
    worker_extra_env: dict[str, str] | None = None,
    task_root_dir: str | Path | None = None,
) -> None:
    self.task_desc = str(task_desc or "")
    self._task_metric_lower_is_better = metric_lower_is_better_hint(self.task_desc)
    self.cfg = cfg
    self.lhr = cfg.lnr
    memory_policy = ensure_lnr_context_memory_budget(self.cfg, self.lhr)
    if memory_policy.changed:
        logger.info(
            "[lnr] raised max_messages from %d to %d (%s)",
            memory_policy.original_max_messages,
            memory_policy.effective_max_messages,
            memory_policy.reason,
        )
    self.orchestrator = orchestrator
    self.worker_id = str(worker_id or "")
    self.worker_index = int(worker_index or 0)
    self.worker_count = max(1, int(worker_count or 1))
    self.worker_extra_env = dict(worker_extra_env or {})
    self.root_dir = Path(cfg.task_workspace_root_dir).resolve()
    self.task_root_dir = (
        Path(task_root_dir).resolve() if task_root_dir is not None else self.root_dir
    )
    self.global_log_dir = self.task_root_dir / "task_logs"
    self.workspace_dir = Path(cfg.workspace_dir).resolve()
    configured_log_dir = Path(cfg.log_dir).resolve()
    self.log_dir = configured_log_dir if self.worker_id else self.global_log_dir
    self.agent_factory = AgentFactory(
        self.orchestrator.create_science_agent,
        trace_sink=self._record_agent_factory_trace,
    )
    self.ledger_filename = self._safe_ledger_filename(self.lhr.ledger_filename)
    self.ledger_path = self.workspace_dir / self.ledger_filename
    self.memory_dir = self.workspace_dir / ".agent_memory"
    self.modules = build_lnr_module_graph(
        root_dir=self.root_dir,
        workspace_dir=self.workspace_dir,
        ledger_path=self.ledger_path,
        module_state_dir=self.log_dir / "module_state",
    )
    self.workspace_service = self.modules.workspace
    self.memory_service = self.modules.memory
    self.estra_service = self.modules.estra
    self.estra_planner = self.modules.estra_planner
    self.estra_archive_store = self.modules.estra_archive
    self.resource_management_service = self.modules.resource_management
    self.execution_value_service = self.modules.execution_value
    self.stage_lifecycle_coordinator = self.modules.stage_lifecycle
    self.finalization_service = self.modules.finalization
    self.prompt_context_builder = self.modules.prompt_context
    self.telemetry_journal = self.modules.telemetry
    self.stage_lifecycle_coordinator.set_trace_sink(self._record_stage_lifecycle_trace)
    self.snapshot_store = SnapshotStore(
        root_dir=self.root_dir,
        workspace_dir=self.workspace_dir,
        snapshot_dirname="snapshots",
        archive_dirname=str(
            getattr(self.lhr, "archive_dirname", "") or "snapshots/archives"
        ),
        control_log_dir=self.log_dir,
        metadata_dirname="logs",
        strict_layout=True,
        memory_dir=self.memory_dir,
        workspace_snapshot_enabled=bool(
            getattr(self.lhr, "workspace_snapshot_enabled", False)
        ),
        workspace_snapshot_verify_objects=bool(
            getattr(self.lhr, "workspace_snapshot_verify_objects", True)
        ),
        task_root_dir=self.task_root_dir,
        worker_id=self.worker_id or "w00",
    )
    _initialize_run_state(self)
    event_observer = _runtime_event_observer(self)
    self.state_machine = LHRStateMachineStore(
        log_dir=self.log_dir,
        worker_id=self.worker_id or "W00",
        worker_index=self.worker_index,
        worker_count=self.worker_count,
        ledger_filename=self.ledger_filename,
        event_observer=event_observer,
    )
    self._resource_arbiter_agent: Any | None = None
    self._resource_admission_agent: Any | None = None
    self._metric_validity_feedback_llm: Any | None = None
    self._resource_main_agent_ref: Any | None = None
    self._live_agent: Any | None = None
    self._merge_owner_solver: LnrSolver | None = None
    self.assessment_pipeline = self.modules.assessment
    self.gate_service = CandidateAssessmentService(
        self.modules.evaluator,
        self.modules.gate,
    )
    # Compatibility for agent hooks and older integrations.
    self.evaluation_service = self.gate_service
    self.evaluator_manager = self.evaluation_service.manager
    self.skill_registry: SkillRegistry | None = None
    self.skill_task_category = ""
    self.skill_allow_names: tuple[str, ...] = ()
    self.skill_tool_mode = str(
        getattr(self.lhr, "lnr_skill_tool_mode", "category_only") or "category_only"
    )
    self.skill_allow_generic_wildcard = bool(
        getattr(self.lhr, "lnr_skill_allow_generic_wildcard", False)
    )
    self.skill_visible_max = int(getattr(self.lhr, "lnr_skill_visible_max", 1) or 1)
    self.skill_category_status = "disabled"
    self._configure_lnr_category_skill()
    self.resource_observer = self._make_resource_observer()


def _initialize_run_state(self) -> None:
    """Initialize mutable run counters owned by the coordinator session."""
    self.deadline = time.monotonic() + max(60, int(self.lhr.wall_clock_budget_sec or 0))
    self.stage_snapshots: dict[str, StageSnapshot] = {}
    self.archived_stage_snapshots: dict[str, StageSnapshot] = {}
    self.duplicate_submission_skip_keys: set[str] = set()
    self.last_captured_solution_sha = ""
    self.initial_workspace_state = ""
    self.last_captured_run_signature = ""
    self.last_stage_commit_ts = 0.0
    self._s01_eda_prefix_end_index: int | None = None
    self._s01_agent_eda_summary = ""
    self.last_estra_stage_count = 0
    self.last_estra_observation_key = ""
    self.last_force_estra_observation_count = 0
    self.last_force_estra_observation_key = ""
    self.last_context_limit_estra_generation = ""
    self.last_context_limit_estra_restore_key = ""
    self.context_limit_estra_restore_keys: set[str] = set()
    self.estra_decisions = 0
    self.current_lineage_no = 1
    self.current_lineage_id = "L01"
    self.current_restored_from_node_uid = ""
    self.pending_estra: dict[str, Any] | None = None
    self.pending_stage_commit_transaction: dict[str, Any] | None = None
    self.evaluator_stop_requested = False
    self.evaluator_stop_reason = ""
    self.main_tokens_in = 0
    self.main_tokens_out = 0
    self.main_tokens_cached = 0
    self.main_llm_calls = 0
    self.context_hygiene_cache_rates: list[float] = []
    self.context_hygiene_last_stage_tokens_in = 0
    self.context_hygiene_last_stage_tokens_cached = 0
    self.context_hygiene_last_compact_tokens_in = 0
    self.context_hygiene_last_compact_stage_count = 0
    self.context_hygiene_last_compact_ts = time.time()
    self.stage_tokens_in = 0
    self.stage_tokens_out = 0
    self.stage_tokens_cached = 0
    self.stage_llm_calls = 0
    self.estra_tokens_in = 0
    self.estra_tokens_out = 0
    self.estra_tokens_cached = 0
    self.estra_llm_calls = 0
    self.current_restored_from_stage = ""
    self.started_at = time.time()


def _runtime_event_observer(self) -> Any | None:
    """Build the optional legacy event projection without owning runtime state."""
    event_observer = None
    if (
        os.environ.get("SCIENCEFLOW_RUNTIME_EVENTS", "").strip()
        or os.environ.get("SCIENCEFLOW_RUNTIME_EVENT_LOG", "").strip()
    ):
        from scienceflow.runtime.events import event_bridge_from_env

        event_bridge = event_bridge_from_env(
            log_dir=self.log_dir,
            worker_id=self.worker_id or "W00",
        )
        event_observer = event_bridge.emit_legacy if event_bridge is not None else None
    return event_observer


def _repo_root(self) -> Path:
    return Path(__file__).resolve().parents[3]


def _skill_library_dir(self) -> Path:
    return default_skill_library_dir(self._repo_root())


def _lnr_skill_category_source_path(self) -> Path:
    configured = str(
        getattr(
            self.lhr,
            "lnr_skill_category_source",
            "tasks/ml/mlebench/competition_categories.json",
        )
        or ""
    ).strip()
    path = Path(configured or "tasks/ml/mlebench/competition_categories.json")
    if path.is_absolute():
        return path
    return self._repo_root() / path


def _load_lnr_task_category_label(self) -> str:
    exp_id = str(getattr(self.cfg, "exp_id", "") or "").strip()
    if not exp_id:
        return ""
    path = self._lnr_skill_category_source_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    label = raw.get(exp_id) if isinstance(raw, dict) else ""
    return str(label or "").strip()


def _configure_lnr_category_skill(self) -> None:
    self.skill_registry = None
    self.skill_task_category = ""
    self.skill_allow_names = ()
    self.skill_category_status = "disabled"
    if not bool(getattr(self.lhr, "lnr_skill_tool_enabled", False)):
        return
    mode = str(
        getattr(self.lhr, "lnr_skill_tool_mode", "category_only") or "category_only"
    ).strip()
    self.skill_tool_mode = mode or "category_only"
    if self.skill_tool_mode != "category_only":
        self.skill_category_status = "unsupported_mode"
        return
    self.skill_allow_generic_wildcard = bool(
        getattr(self.lhr, "lnr_skill_allow_generic_wildcard", False)
    )
    self.skill_visible_max = max(
        1, int(getattr(self.lhr, "lnr_skill_visible_max", 1) or 1)
    )
    label_field = str(
        getattr(self.lhr, "lnr_skill_category_label_field", "category_label")
        or "category_label"
    ).strip()
    if label_field != "category_label":
        self.skill_category_status = "unsupported_category_label_field"
        return
    label = self._load_lnr_task_category_label()
    self.skill_task_category = label
    if not label:
        self.skill_category_status = "skill_category_missing"
        return
    registry = getattr(self.orchestrator, "skill_registry", None)
    if registry is None:
        registry = SkillRegistry()
        try:
            registry.load_all(self._skill_library_dir())
        except Exception:
            self.skill_category_status = "skill_registry_load_failed"
            return
    candidates = registry.get_by_category_label(label)
    if not candidates:
        self.skill_category_status = "skill_category_unmapped"
        return
    selected = candidates[: self.skill_visible_max]
    selected_names: list[str] = []
    for skill in selected:
        name = skill.metadata.name
        if name and name not in selected_names:
            selected_names.append(name)
    exp_id = str(getattr(self.cfg, "exp_id", "") or "").strip()
    if exp_id:
        try:
            task_skill = registry.get_by_name_or_alias(exp_id)
        except Exception:
            task_skill = None
        if (
            task_skill is not None
            and task_skill.metadata.category == "tasks"
            and task_skill.metadata.name
            and task_skill.metadata.name not in selected_names
        ):
            selected_names.append(task_skill.metadata.name)
    self.skill_registry = registry
    self.skill_allow_names = tuple(selected_names)
    self.skill_visible_max = max(self.skill_visible_max, len(self.skill_allow_names))
    self.skill_category_status = (
        "enabled" if self.skill_allow_names else "skill_category_unmapped"
    )


@staticmethod
def _extract_lnr_auto_skill_hint(rendered: str) -> str:
    """Return a short auto-load hint section when a skill provides one."""
    lines = str(rendered or "").splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        normalized = line.strip().lower().replace("_", "-")
        if normalized in {
            "## auto-load hint",
            "## auto-load summary",
            "## auto load hint",
        }:
            start = idx + 1
            break
    if start is None:
        return str(rendered or "").strip()
    collected: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


@staticmethod
def _lnr_compact_skill_rendered(meta: Any, rendered: str, *, max_chars: int) -> str:
    text = str(rendered or "").strip()
    if not text:
        return ""
    if str(getattr(meta, "category", "") or "") != "categories":
        return text
    if len(text) <= max_chars:
        return text
    excerpt = text[: max(200, max_chars)].rstrip()
    return (
        excerpt
        + "\n\n[category excerpt only; use `skill read` for the full category skill]"
    )


def _lnr_auto_read_skill_block(self) -> str:
    if not bool(getattr(self.lhr, "lnr_skill_auto_read", False)):
        return ""
    if self.skill_registry is None or not self.skill_allow_names:
        return ""
    max_chars = max(
        500, int(getattr(self.lhr, "lnr_skill_auto_read_max_chars", 6000) or 6000)
    )
    category_max_chars = max(
        400,
        int(getattr(self.lhr, "lnr_skill_category_auto_read_max_chars", 800) or 800),
    )
    chunks: list[str] = []
    remaining = max_chars
    skill_entries: list[Any] = []
    for name in self.skill_allow_names:
        try:
            skill = self.skill_registry.get_by_name_or_alias(name)
            if skill is None and hasattr(self.skill_registry, "get_by_name"):
                skill = self.skill_registry.get_by_name(name)
        except Exception:
            skill = None
        if skill is None:
            continue
        skill_entries.append(skill)
    skill_entries.sort(
        key=lambda skill: (
            0 if str(getattr(skill.metadata, "category", "") or "") == "tasks" else 1
        )
    )
    for skill in skill_entries:
        meta = skill.metadata
        tags = ",".join(getattr(meta, "tags", []) or []) or "-"
        rendered = str(skill.render() or "").strip()
        auto_hint = self._extract_lnr_auto_skill_hint(rendered)
        compacted = self._lnr_compact_skill_rendered(
            meta,
            auto_hint,
            max_chars=category_max_chars,
        )
        if not compacted:
            continue
        if compacted != auto_hint:
            source = "category_excerpt"
        elif auto_hint != rendered:
            source = "auto_hint"
        else:
            source = "full_compact"
        block = (
            f"[skill:{meta.name}]\n"
            f"category={getattr(meta, 'category', '') or '-'} tags={tags} source={source}\n"
            f"{compacted}\n"
        )
        if len(block) > remaining:
            if remaining < 200:
                break
            chunks.append(block[: remaining - 3].rstrip() + "...")
            break
        chunks.append(block)
        remaining -= len(block)
    if not chunks:
        return ""
    return (
        "\nAuto-loaded compact skill hints from enabled category/task skills "
        "because lnr_skill_auto_read=true. Use `skill read <name>` for full guidance:\n"
        + "\n".join(chunks).strip()
        + "\n"
    )


def _lnr_skill_hint(self) -> str:
    if self.skill_registry is None or not self.skill_allow_names:
        return ""
    if any(name.startswith("task_") for name in self.skill_allow_names):
        base = "Category- and task-specific skills are available. Use `skill list` if you want guidance for this task."
    else:
        base = "A category-specific skill is available. Use `skill list` if you want guidance for this task type."
    return base + self._lnr_auto_read_skill_block()


@staticmethod
def _safe_ledger_filename(raw: str) -> str:
    name = str(raw or ".run_results.md").strip().replace("\\", "/")
    if not name or "/" in name or name in {".", ".."}:
        return ".run_results.md"
    if name.startswith(".") and name != ".run_results.md":
        return ".run_results.md"
    return name
