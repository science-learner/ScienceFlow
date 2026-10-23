# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Static gates for the V3.1 core decomposition."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib
from types import SimpleNamespace

import yaml

from scienceflow.foundation.config.schema.settings import Config
from scienceflow.research.solver.lnr.orchestration.solver import LnrSolver


PACKAGE_ROOT = Path(__file__).parents[1] / "scienceflow"
COORDINATOR_ROOT = PACKAGE_ROOT / "research/solver/lnr/orchestration/coordinator"
OBSERVER_ROOT = PACKAGE_ROOT / "research/solver/lnr/resources/runtime/observer"
RESOURCE_RUNTIME_COMPONENTS = (
    PACKAGE_ROOT / "research/solver/lnr/resources/runtime/runtime_components"
)
AGENT_COMPONENTS = PACKAGE_ROOT / "agent/components"
RUN_LOOP_COMPONENTS = PACKAGE_ROOT / "agent/runtime/run_loop_components"
MEMORY_CONTEXT_COMPONENTS = PACKAGE_ROOT / "research/state/knowledge/context/context"
BASH_COMPONENTS = PACKAGE_ROOT / "runtime/safety/tooling/bash"
CONFIG_COMPONENTS = PACKAGE_ROOT / "foundation/config/schema/models"
MONITOR_COMPONENTS = PACKAGE_ROOT / "interfaces/ui/monitor/lnr"
PARALLEL_COMPONENTS = PACKAGE_ROOT / "runtime" / "parallel"
LARGE_OBJECT_ALLOWLIST = (
    PACKAGE_ROOT / "foundation/architecture/large_object_allowlist.yaml"
)
RESPONSIBILITY_ROOTS = (
    COORDINATOR_ROOT,
    OBSERVER_ROOT,
    RESOURCE_RUNTIME_COMPONENTS,
    AGENT_COMPONENTS,
    RUN_LOOP_COMPONENTS,
    MEMORY_CONTEXT_COMPONENTS,
    BASH_COMPONENTS,
    CONFIG_COMPONENTS,
    MONITOR_COMPONENTS,
    PARALLEL_COMPONENTS,
)


def _function_names(root: Path, *, excluded: set[str]) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return names


def _bound_names(
    path: Path,
    class_name: str,
    *,
    ignored: set[str] | None = None,
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in klass.body
    )
    ignored_names = {"solver_name"} if ignored is None else ignored
    return {
        target.id
        for node in klass.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id not in ignored_names
    }


def _owned_method_names(root: Path, *, excluded: set[str]) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for klass in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            names.update(
                node.name
                for node in klass.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return names


def _composition_bases(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign))
        for node in klass.body
    )
    return {base.id for base in klass.bases if isinstance(base, ast.Name)}


def test_removed_core_files_do_not_exist() -> None:
    assert not (PACKAGE_ROOT / "research/solver/lnr/solver_core.py").exists()
    assert not (OBSERVER_ROOT / "observer_core.py").exists()


def test_coordinator_facade_binds_every_moved_responsibility() -> None:
    moved = _function_names(
        COORDINATOR_ROOT,
        excluded={"__init__.py", "shared.py", "solver.py"},
    )
    bound = _bound_names(COORDINATOR_ROOT / "solver.py", "LnrSolver")
    assert len(moved) == 257
    assert len(bound) == 234
    assert bound < moved


def test_coordinator_constructor_wires_multi_worker_merge_services(
    tmp_path: Path,
) -> None:
    cfg = Config()
    cfg.task_workspace_root_dir = str(tmp_path)
    cfg.workspace_dir = str(tmp_path / "workspace")
    cfg.log_dir = str(tmp_path / "logs")
    cfg.lnr.resource_monitor_enabled = False
    orchestrator = SimpleNamespace(create_science_agent=lambda **_kwargs: None)

    solver = LnrSolver(
        task_desc="constructor characterization", cfg=cfg, orchestrator=orchestrator
    )

    assert solver.assessment_pipeline is solver.modules.assessment
    assert solver.evaluation_service is solver.gate_service
    assert solver.evaluator_manager is solver.evaluation_service.manager
    assert solver.resource_observer is None


def test_observer_composes_explicit_responsibility_owners() -> None:
    owned = _owned_method_names(
        OBSERVER_ROOT,
        excluded={
            "__init__.py",
            "shared.py",
            "controller.py",
            "construction_runtime.py",
        },
    )
    bases = _composition_bases(OBSERVER_ROOT / "controller.py", "LHRResourceObserver")
    assert len(owned) == 233
    assert len(bases) == 22
    assert not (OBSERVER_ROOT / "observer.py").exists()


def test_new_responsibility_files_respect_size_gate() -> None:
    excluded = {"__init__.py", "shared.py", "solver.py", "observer.py", "controller.py"}
    oversized = {
        str(path.relative_to(PACKAGE_ROOT)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for root in (COORDINATOR_ROOT, OBSERVER_ROOT)
        for path in root.rglob("*.py")
        if path.name not in excluded
        and len(path.read_text(encoding="utf-8").splitlines()) > 1_000
    }
    assert oversized == {}


def test_responsibility_modules_do_not_retain_host_back_references() -> None:
    forbidden_attributes = {"host", "_host", "owner", "_owner", "solver", "_solver"}
    violations: list[str] = []
    for root in (COORDINATOR_ROOT, OBSERVER_ROOT):
        for path in root.glob("*.py"):
            if path.name in {"shared.py", "solver.py", "observer.py", "controller.py"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr in forbidden_attributes
                    ):
                        violations.append(
                            f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}"
                        )
    assert violations == []


def test_resource_runtime_composes_explicit_responsibility_owners() -> None:
    owned = _owned_method_names(
        RESOURCE_RUNTIME_COMPONENTS,
        excluded={"__init__.py", "shared.py"},
    )
    bases = _composition_bases(
        PACKAGE_ROOT / "research/solver/lnr/resources/runtime/execution/facade.py",
        "ResourceRuntime",
    )
    assert len(owned) == 76
    assert len(bases) == 7
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in RESOURCE_RUNTIME_COMPONENTS.rglob("*.py")
    )


def test_science_agent_facade_uses_typed_host_ports() -> None:
    owner_files = (
        PACKAGE_ROOT / "agent/factory/construction/__init__.py",
        PACKAGE_ROOT / "research/state/workspace/adapters/path_hygiene.py",
        PACKAGE_ROOT / "research/state/workspace/session/agent_session.py",
        PACKAGE_ROOT / "runtime/observability/agent_tool_feedback.py",
        PACKAGE_ROOT / "research/control/agent_runtime_state.py",
        PACKAGE_ROOT / "runtime/safety/execution/agent_runtime/edit_guard.py",
        PACKAGE_ROOT / "research/state/knowledge/memory/context/agent_compaction.py",
    )
    moved = {
        node.name
        for path in owner_files
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bound = _bound_names(
        PACKAGE_ROOT / "agent/core/runtime/agent.py",
        "ScienceAgent",
        ignored={"max_steps", "model_config"},
    )
    moved.update(
        {
            "run",
            "_agentic_route_log_dir",
            "_write_agentic_route_response",
            "_build_lnr_stage_commit_compact_messages",
            "_sync_resource_state_summary_slot",
            "_append_agentic_route_decision_log",
            "_accumulate_last_llm_call_tokens_into_run",
            "run_ephemeral_agentic_route_prompt",
            "_looks_like_repl_small_talk",
            "_tool_choice_for_main_loop",
            "_unavailable_repl_tool_names",
            "_tc_function_name",
            "_tc_arguments_dict",
            "_repl_file_change_command_from_tool",
            "_normalize_repl_file_change_tool_calls",
            "_block_unavailable_repl_tool_calls",
            "_normalize_tool_calls_for_execution",
            "_add_message_after_current_tool_bundle",
            "_ask_tool_stream_guarded",
            "_record_llm_call",
            "_build_system_prompt",
            "_build_system_messages",
            "_format_round_budget",
            "_maybe_append_solution_write_nudge",
            "_maybe_append_write_failure_coaching",
            "_solution_write_syntax_or_short_failure",
            "_try_write_solution_from_assistant_text",
            "_lnr_maybe_fresh_workspace_hint_first_round",
            "_lnr_maybe_periodic_inject_at_round_start",
            "_lnr_on_round_complete",
            "_maybe_log_lnr_llm_turn",
            "_maybe_log_sft_turn",
            "_lnr_get_last_assistant_text",
            "_lnr_should_attach_search_phase_hard_constraint_reminder",
            "_lnr_guard_inject_search_phase_suffix",
            "_lnr_maybe_build_search_phase_constraint_correction",
            "_lnr_maybe_inject_requested_skill",
            "_maybe_write_bare_run_tail_snapshot",
            "_inject_run_control_user_message",
            "_append_to_last_tool_message",
            "_inject_run_control_ok_message",
            "_mirror_embedded_full_run_to_interaction_log",
            "_append_embedded_full_run_to_result_md",
            "_append_mlebench_validation_to_result_md",
            "_run_mlebench_validation_after_embedded_full_run",
            "_maybe_embedded_full_run_after_quick_test",
        }
    )
    assert len(moved) == 107
    migrated_to_ports = {
        "_maybe_workspace_git_auto_checkpoint",
        "_maybe_normalize_tool_input_paths",
        "_maybe_rewrite_tool_result_paths",
        "_execute_tool_maybe_edit_guard",
        "_record_read_for_edit_guard",
        "_prepare_tool_feedback_for_memory",
        "_log_iteration_header",
        "_pop_and_log_thought",
        "_lnr_maybe_fresh_workspace_hint_first_round",
        "_lnr_maybe_periodic_inject_at_round_start",
        "_lnr_on_round_complete",
        "_sync_resource_state_summary_slot",
        "_write_productivity_snapshot",
        "_build_system_messages",
        "_tool_choice_for_main_loop",
        "_normalize_repl_file_change_tool_calls",
        "_normalize_tool_calls_for_execution",
        "_maybe_write_bare_run_tail_snapshot",
        "_maybe_embedded_full_run_after_quick_test",
    }
    retired_legacy_methods = {
        "_add_assistant_api_message",
        "_next_step",
        "_lnr_invalidate_current_valid_run",
        "_lnr_adjust_policy_injection",
        "seed_edit_guard",
        "_parent_solution_path",
        "classify_tool_error_for_budget",
        "_maybe_log_lnr_llm_turn",
        "_maybe_log_sft_turn",
        "_lnr_get_last_assistant_text",
        "_lnr_should_attach_search_phase_hard_constraint_reminder",
        "_lnr_guard_inject_search_phase_suffix",
        "_lnr_maybe_build_search_phase_constraint_correction",
        "_lnr_maybe_inject_requested_skill",
    }
    assert len(bound) == 68
    assert bound < moved
    # Coordinator stage adapters still consume this one public compatibility
    # method; the session itself now uses the typed port.
    shared_compatibility = {
        "_build_system_messages",
        "_lnr_maybe_periodic_inject_at_round_start",
        "_maybe_write_bare_run_tail_snapshot",
        "_prepare_tool_feedback_for_memory",
        "_sync_resource_state_summary_slot",
    }
    assert (migrated_to_ports - shared_compatibility).isdisjoint(bound)
    assert shared_compatibility <= bound
    assert retired_legacy_methods.isdisjoint(bound)
    assert (migrated_to_ports | retired_legacy_methods) <= moved
    assert (PACKAGE_ROOT / "agent/core/ports/host_ports.py").is_file()
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in owner_files
    )


def test_legacy_run_loop_facade_and_components_are_removed() -> None:
    assert not (PACKAGE_ROOT / "agent/runtime/run_loop.py").exists()
    retired = {
        "loop_completion.py",
        "model_turn.py",
        "round_loop.py",
        "tool_dispatch.py",
        "turn_handling.py",
    }
    assert not retired & {path.name for path in RUN_LOOP_COMPONENTS.glob("*.py")}
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in RUN_LOOP_COMPONENTS.glob("*.py")
    )


def test_memory_context_facade_binds_state_owner_and_keeps_files_bounded() -> None:
    manager_files = {
        "runtime/manager_state.py",
        "projection/manager_projection.py",
        "compression/manager_compaction.py",
        "runtime/manager_tool_results.py",
        "runtime/manager_finalize.py",
    }
    moved: set[str] = set()
    for filename in manager_files:
        tree = ast.parse(
            (MEMORY_CONTEXT_COMPONENTS / filename).read_text(encoding="utf-8")
        )
        moved.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    bound = _bound_names(
        PACKAGE_ROOT / "research/state/knowledge/context/memory_context.py",
        "MemoryContextManager",
        ignored=set(),
    )
    assert len(moved) == 26
    assert bound == moved
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in MEMORY_CONTEXT_COMPONENTS.rglob("*.py")
    )


def test_bash_tool_is_a_facade_over_policy_and_process_components() -> None:
    tool_path = BASH_COMPONENTS / "core/tool.py"
    tree = ast.parse(tool_path.read_text(encoding="utf-8"), filename=str(tool_path))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BashTool"
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in klass.body
    )
    bound = {
        target.id
        for node in klass.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {
        "_apply_hard_fuse_timeout",
        "_deadline_state",
        "execute",
        "_execute_with_stream_monitoring",
        "_execute_process_pipeline",
    } <= bound
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in BASH_COMPONENTS.glob("*.py")
        if path.name != "process_pipeline.py"
    )


def test_config_and_monitor_components_are_bounded() -> None:
    for root in (CONFIG_COMPONENTS, MONITOR_COMPONENTS):
        assert all(
            len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
            for path in root.glob("*.py")
        )
    assert (
        len(
            (PACKAGE_ROOT / "foundation/config/schema/settings.py")
            .read_text()
            .splitlines()
        )
        <= 100
    )
    assert (
        len(
            (PACKAGE_ROOT / "interfaces/ui/monitor/lnr_state.py")
            .read_text()
            .splitlines()
        )
        <= 100
    )


def test_parallel_runner_facade_binds_every_extracted_responsibility() -> None:
    component_files = {
        "execution/construction.py",
        "execution/scheduler.py",
        "execution/subprocess.py",
        "state/resume.py",
        "state/lifecycle.py",
        "state/result.py",
    }
    moved: set[str] = set()
    for filename in component_files:
        tree = ast.parse((PARALLEL_COMPONENTS / filename).read_text(encoding="utf-8"))
        moved.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    bound = _bound_names(
        PARALLEL_COMPONENTS / "execution/runner.py",
        "ParallelRunner",
        ignored=set(),
    )
    assert len(moved) == 16
    assert bound == moved
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
        for path in PARALLEL_COMPONENTS.rglob("*.py")
    )


def test_planned_large_test_suites_are_split_without_large_parts() -> None:
    stems = (
        "long_horizon_repl",
        "resource_arbiter_control_plane",
        "science_agent_repl",
        "tool_memory_compression",
    )
    tests_root = PACKAGE_ROOT.parent / "tests"
    for stem in stems:
        assert not (tests_root / f"test_{stem}.py").exists()
        parts = list(tests_root.glob(f"test_{stem}_*.py"))
        assert parts
        assert all(
            len(path.read_text(encoding="utf-8").splitlines()) <= 1_000
            for path in parts
        )


def test_large_object_allowlist_exactly_covers_current_exceptions() -> None:
    document = yaml.safe_load(LARGE_OBJECT_ALLOWLIST.read_text(encoding="utf-8"))
    policy = document["policy"]
    assert policy["max_function_lines"] == 150
    assert policy["reason"]
    assert policy["expiry"]
    assert policy["delete_when"]

    registered = {item["object"] for item in document["exceptions"]}
    assert len(registered) == len(document["exceptions"])
    assert all(item["kind"] and item["owner"] for item in document["exceptions"])

    discovered: set[str] = set()
    for root in RESPONSIBILITY_ROOTS:
        for path in root.rglob("*.py"):
            module = ".".join(
                path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
            )
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                assert node.end_lineno is not None
                if node.end_lineno - node.lineno + 1 > policy["max_function_lines"]:
                    discovered.add(f"{module}:{node.name}")

    assert registered == discovered


def test_distribution_declares_runtime_packages_and_default_config_data() -> None:
    assert (PACKAGE_ROOT / "agent/policies/__init__.py").is_file()
    pyproject = tomllib.loads(
        (PACKAGE_ROOT.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    config_data = package_data["scienceflow.foundation.config"]
    profile_data = package_data["scienceflow.foundation.config.profiles"]
    assert {
        "default.yaml",
        "sci_modeling_bench.yaml",
    } <= set(config_data)
    assert {
        "defaults/*.yaml",
        "defaults/lnr/*.yaml",
    } <= set(profile_data)
