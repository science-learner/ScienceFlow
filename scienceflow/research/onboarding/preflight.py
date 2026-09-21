"""Structural preflight using existing dataset, Parallel, evaluator, and gate owners."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml

from scienceflow.foundation.contracts import EvalContext
from scienceflow.research.onboarding.session import (
    parse_cpu_list,
    parse_gpu_selection,
    parse_workers,
)
from scienceflow.research.onboarding.support.evaluator_probe import run_evaluator_probe
from scienceflow.research.onboarding.support.gate_probe import run_gate_probe
from scienceflow.research.quality.evaluator.manager import EvaluatorManager
from scienceflow.research.state.dataset.discovery.scan import scan_data_dir
from scienceflow.runtime.core.support.system_resources import (
    parse_cpu_list as expand_cpu_list,
)
from scienceflow.runtime.parallel.execution.runner import ParallelRunner
from scienceflow.runtime.task_package import find_task_package


class PreflightStatus(str, Enum):
    VERIFIED = "verified"
    EXPLORATORY = "exploratory"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    status: PreflightStatus
    manifest_path: str
    checks: tuple[PreflightCheck, ...]
    task_id: str = ""
    evaluator_backend: str = ""
    report_path: str = ""

    @property
    def ok(self) -> bool:
        return self.status != PreflightStatus.FAILED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": self.status.value,
            "manifest_path": self.manifest_path,
            "task_id": self.task_id,
            "evaluator_backend": self.evaluator_backend,
            "checks": [asdict(check) for check in self.checks],
        }


def run_preflight(
    manifest_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> PreflightReport:
    """Validate an onboarding manifest without executing user or evaluator code.

    A registered, authoritative TaskPackage can be ``verified`` structurally.
    An otherwise valid artifact-command task remains ``exploratory`` until a
    domain owner supplies an authoritative package. Any hard check failure is
    ``failed`` and must not be launched implicitly.
    """

    path = Path(manifest_path).expanduser().resolve(strict=False)
    destination = (
        Path(report_path)
        if report_path is not None
        else path.with_name("preflight_report.json")
    )
    checks: list[PreflightCheck] = []
    task_id = ""
    backend_name = ""
    registered_authoritative = False

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be a mapping")
        tasks = payload.get("tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != 1
            or not isinstance(tasks[0], dict)
        ):
            raise ValueError(
                "onboarding manifest must contain exactly one task mapping"
            )
        task = tasks[0]
        checks.append(
            PreflightCheck("manifest_shape", True, "one canonical Parallel task")
        )
    except Exception as exc:
        checks.append(
            PreflightCheck("manifest_shape", False, f"{type(exc).__name__}: {exc}")
        )
        return _finish(PreflightStatus.FAILED, path, destination, checks)

    try:
        ParallelRunner(path)
        checks.append(
            PreflightCheck("parallel_loader", True, "accepted by ParallelRunner loader")
        )
    except Exception as exc:
        checks.append(
            PreflightCheck("parallel_loader", False, f"{type(exc).__name__}: {exc}")
        )

    lnr = task.get("lnr")
    lnr = lnr if isinstance(lnr, dict) else {}
    try:
        cpu_list = parse_cpu_list(str(task.get("cpu_list") or ""))
        gpu_list = parse_gpu_selection(str(task.get("gpu_list") or ""))
        workers = parse_workers(lnr.get("num_workers", 0))
        cpu_count = len(expand_cpu_list(cpu_list))
        if cpu_count < workers:
            raise ValueError(
                f"CPU pool has {cpu_count} cores for {workers} workers; each worker needs one"
            )
        checks.append(
            PreflightCheck(
                "resource_config",
                True,
                f"cpu={cpu_list}; gpu={gpu_list}; workers={workers}",
            )
        )
    except (TypeError, ValueError) as exc:
        checks.append(PreflightCheck("resource_config", False, str(exc)))

    task_id = str(task.get("exp_id") or "").strip()
    input_dir = Path(str(task.get("input_data_dir") or "")).expanduser()
    try:
        scan = scan_data_dir(
            input_dir, walk_budget_dirs=2_000, walk_budget_files=10_000
        )
        scan_error = str(scan.get("error") or "")
        checks.append(
            PreflightCheck(
                "dataset_scan",
                not scan_error,
                scan_error
                or f"readable; {scan.get('stats', {}).get('total_files', 0)} files sampled",
            )
        )
    except Exception as exc:
        checks.append(
            PreflightCheck("dataset_scan", False, f"{type(exc).__name__}: {exc}")
        )

    evaluator = task.get("evaluator")
    evaluator = evaluator if isinstance(evaluator, dict) else {}
    backend_name = str(evaluator.get("backend") or "").strip()
    manager = EvaluatorManager.default()
    backend = manager.get(backend_name)
    checks.append(
        PreflightCheck(
            "evaluator_registry",
            backend is not None,
            f"registered backend: {backend_name}"
            if backend is not None
            else f"unknown backend: {backend_name or '<empty>'}",
        )
    )

    candidate = evaluator.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    metric = evaluator.get("metric")
    metric = metric if isinstance(metric, dict) else {}
    command = evaluator.get("command")
    command = command if isinstance(command, dict) else {}
    artifact_path = str(candidate.get("artifact") or "").strip()
    metric_name = str(metric.get("name") or "").strip()
    lower_is_better = metric.get("lower_is_better")
    direction_deferred = bool(
        backend_name == "task_package"
        and str(evaluator.get("task_profile") or "").strip().casefold()
        == "mlebench"
        and lower_is_better is None
    )
    contract_ok = bool(
        artifact_path
        and metric_name
        and (isinstance(lower_is_better, bool) or direction_deferred)
    )
    detail = (
        "artifact and metric are explicit; direction resolves from runtime evidence"
        if direction_deferred
        else "artifact, metric, and direction are explicit"
    )
    if not contract_ok:
        detail = (
            "artifact path, metric name, and boolean direction are required; "
            "registered MLEBench tasks may defer direction"
        )
    checks.append(PreflightCheck("evaluation_contract", contract_ok, detail))

    command_ok = True
    if backend_name == "artifact_command":
        command_ok = bool(str(command.get("evaluator_command") or "").strip())
        checks.append(
            PreflightCheck(
                "artifact_command",
                command_ok,
                "trusted evaluator command configured"
                if command_ok
                else "artifact_command backend requires evaluator_command",
            )
        )

    workspace_root = (
        path.parent.parent if path.parent.name == ".scienceflow" else path.parent
    )
    evaluator_context = EvalContext(
        task_profile=str(evaluator.get("task_profile") or "auto"),
        task_id=task_id,
        task_root=workspace_root,
        workspace=workspace_root,
        worker_id="preflight",
        stage_id="preflight",
        cfg={"evaluator": evaluator, "gate": task.get("gate") or {}},
    )
    if backend is not None:
        try:
            selected_backend = manager.backend_name_for_context(evaluator_context)
            backend.prepare_workspace(evaluator_context)
            prompt_contract = backend.build_prompt_contract(evaluator_context).strip()
            backend.detect_candidates(evaluator_context)
            smoke_ok = selected_backend == backend_name and bool(prompt_contract)
            checks.append(
                PreflightCheck(
                    "evaluator_smoke",
                    smoke_ok,
                    "backend selection, prompt contract, and candidate detection completed"
                    if smoke_ok
                    else f"configured={backend_name}; selected={selected_backend or '<empty>'}",
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck("evaluator_smoke", False, f"{type(exc).__name__}: {exc}")
            )

    try:
        from scienceflow.runtime.task_package import load_task_package
        source = evaluator.get('package_source')
        package = load_task_package(Path(source)) if source else find_task_package(task_id)
    except Exception as exc:
        package = None
        checks.append(
            PreflightCheck("task_package", False, f"{type(exc).__name__}: {exc}")
        )
    if package is not None and backend_name == "task_package":
        package_matches = (
            package.metric_name == metric_name
            and (package.lower_is_better is None or package.lower_is_better is lower_is_better)
            and package.artifact_path == artifact_path
            and bool(package.evaluator_entrypoint)
        )
        registered_authoritative = bool(
            package_matches and package.metric_authoritative
        )
        checks.append(
            PreflightCheck(
                "task_package",
                package_matches,
                "registered TaskPackage contract matches manifest"
                if package_matches
                else "manifest diverges from its registered TaskPackage",
            )
        )
        if package_matches and backend is not None:
            try:
                probe = run_evaluator_probe(backend, evaluator_context, package)
                if probe is not None:
                    checks.append(PreflightCheck("evaluator_execution", *probe))
            except Exception as exc:
                checks.append(
                    PreflightCheck(
                        "evaluator_execution",
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    elif backend_name == "task_package":
        checks.append(
            PreflightCheck(
                "task_package", False, "task_package backend has no registered task"
            )
        )

    gate = task.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    # The synthetic gate probe needs a concrete direction even when the real
    # MLEBench run will resolve it from task and stage evidence.
    probe_lower_is_better = (
        lower_is_better
        if isinstance(lower_is_better, bool)
        else True if direction_deferred else None
    )
    try:
        gate_ok, gate_detail = run_gate_probe(
            task_id=task_id,
            workspace=workspace_root,
            evaluator=evaluator,
            gate=gate,
            backend_name=backend_name,
            artifact_path=artifact_path,
            metric_name=metric_name,
            lower_is_better=probe_lower_is_better,
        )
        checks.append(PreflightCheck("gate_fail_closed", gate_ok, gate_detail))
    except Exception as exc:
        checks.append(
            PreflightCheck("gate_fail_closed", False, f"{type(exc).__name__}: {exc}")
        )

    failed = any(not check.ok for check in checks)
    status = (
        PreflightStatus.FAILED
        if failed
        else PreflightStatus.VERIFIED
        if registered_authoritative
        else PreflightStatus.EXPLORATORY
    )
    return _finish(
        status, path, destination, checks, task_id=task_id, backend_name=backend_name
    )


def _finish(
    status: PreflightStatus,
    manifest_path: Path,
    report_path: Path,
    checks: list[PreflightCheck],
    *,
    task_id: str = "",
    backend_name: str = "",
) -> PreflightReport:
    report = PreflightReport(
        status=status,
        manifest_path=str(manifest_path),
        checks=tuple(checks),
        task_id=task_id,
        evaluator_backend=backend_name,
        report_path=str(report_path.resolve(strict=False)),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["PreflightCheck", "PreflightReport", "PreflightStatus", "run_preflight"]
