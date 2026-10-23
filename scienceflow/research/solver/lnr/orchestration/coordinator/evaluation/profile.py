# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Extracted coordinator component with an explicit dependency surface."""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    EvalContext,
    EvaluationRequest,
    Path,
    SimpleNamespace,
    find_task_package,
    logger,
    merge_adjudicated_stage_facts,
    merge_primary_stage_facts,
    metric_event_to_stage_facts,
    task_command_env,
    task_python_executable,
    time,
)


class EvaluationProfileOwner:
    """Own evaluator configuration and task-runtime projection."""

    def _evaluator_event_log_name(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        raw = str(
            getattr(evaluator, "event_log", "evaluator_events.jsonl")
            or "evaluator_events.jsonl"
        ).strip()
        if not raw or "/" in raw or "\\" in raw or raw.startswith("."):
            return "evaluator_events.jsonl"
        return raw

    def _evaluator_task_profile(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        return str(
            getattr(evaluator, "task_profile", "")
            or getattr(cfg, "task_profile", "auto")
            or "auto"
        )

    def _code_organization_hint(self) -> str:
        hint = str(getattr(self.lhr, "code_organization_hint", "") or "").strip()
        profile = self._evaluator_task_profile().strip().lower().replace("-", "_")
        if profile in {"opt_solver", "optimization_solver", "artifact_solver"}:
            default_hints = {"", "beyond_mfiles", "beyond_multifile"}
            if hint.lower().replace("-", "_").replace(" ", "_") in default_hints:
                return "opt_solver"
        return hint

    def _evaluator_candidate_artifact(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        candidate = getattr(evaluator, "candidate", None)
        artifact = str(getattr(candidate, "artifact", "") or "").strip()
        spec = self._task_package_spec()
        if (
            spec is not None
            and self._evaluator_backend_name() == "task_package"
            and artifact in {"", "submission.csv"}
        ):
            return spec.artifact_path
        return artifact

    def _evaluator_candidate_artifact_kind(self) -> str:
        evaluator = getattr(self.cfg, "evaluator", None)
        candidate = getattr(evaluator, "candidate", None)
        kind = str(getattr(candidate, "artifact_kind", "") or "").strip()
        spec = self._task_package_spec()
        if (
            spec is not None
            and self._evaluator_backend_name() == "task_package"
            and kind in {"", "submission_csv"}
        ):
            return spec.artifact_kind
        return kind

    def _task_package_spec(self) -> Any | None:
        try:
            return find_task_package(str(getattr(self.cfg, "exp_id", "") or ""))
        except Exception:
            logger.debug("[lnr] task package lookup failed", exc_info=True)
            return None

    def _evaluator_backend_name(self) -> str:
        evaluator = getattr(self.cfg, "evaluator", None)
        configured = str(getattr(evaluator, "backend", "") or "").strip()
        if configured.lower() not in {"", "auto"}:
            return configured
        manager = getattr(self, "evaluator_manager", None)
        if manager is None:
            return ""
        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                cfg=self.cfg,
            )
            return str(manager.backend_name_for_context(ctx) or "").strip()
        except Exception:
            logger.debug("[lnr] evaluator backend resolution failed", exc_info=True)
            return ""

    def _task_runtime_context(self) -> Any:
        return SimpleNamespace(cfg=self.cfg, task_python_executable="")

    def _task_python_executable(self) -> Path | None:
        try:
            return task_python_executable(self._task_runtime_context())
        except Exception:
            logger.debug("[lnr] task python resolution failed", exc_info=True)
            return None

    def _task_runtime_extra_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "SCIENCEFLOW_TASK_PROFILE": self._evaluator_task_profile(),
            "SCIENCEFLOW_EVALUATOR_BACKEND": self._evaluator_backend_name(),
            "SCIENCEFLOW_CANDIDATE_ARTIFACT": self._evaluator_candidate_artifact(),
        }
        artifact_kind = self._evaluator_candidate_artifact_kind()
        if artifact_kind:
            env["SCIENCEFLOW_CANDIDATE_ARTIFACT_KIND"] = artifact_kind
        python_path = self._task_python_executable()
        if not python_path:
            return env
        env["SCIENCEFLOW_TASK_PYTHON"] = str(python_path)
        try:
            selected_env = task_command_env(
                self._task_runtime_context(), python=python_path
            )
        except Exception:
            logger.debug("[lnr] task command environment failed", exc_info=True)
            selected_env = {}
        for key in ("PATH", "CONDA_PREFIX", "VIRTUAL_ENV"):
            value = str(selected_env.get(key) or "").strip()
            if value:
                env[key] = value
        return env

    def _task_runtime_prompt_contract(self) -> str:
        lines: list[str] = []
        artifact = self._evaluator_candidate_artifact()
        if artifact:
            lines.append(f"- Candidate artifact path: `{artifact}`.")
            lines.append(
                "- Stage capture and resume use this artifact path as the deliverable signal."
            )
        python_path = self._task_python_executable()
        if python_path:
            lines.extend(
                [
                    f"- Task Python executable: `{python_path}`.",
                    "- In bash commands, use `$SCIENCEFLOW_TASK_PYTHON` for task dependencies; `python` is also pointed at this environment.",
                    "- Do not install task dependencies into the controller `.venv` or a new ad-hoc workspace environment unless this configured Python fails.",
                ]
            )
        return "\n".join(lines)

    def _workspace_state_deliverable_line(self) -> str:
        artifact = self._evaluator_candidate_artifact()
        profile = self._evaluator_task_profile().strip().lower()
        if profile in {"", "default", "mlebench"} and artifact == "submission.csv":
            return "A valid run must create root-level submission.csv and print Final Validation Score: <float>."
        return f"A valid run must create `{artifact}` and let the configured evaluator report the authoritative metric."

    def _evaluator_prompt_contract(self) -> str:
        service = getattr(self, "assessment_pipeline", None)
        if service is None:
            service = getattr(self, "gate_service", None) or getattr(
                self, "evaluation_service", None
            )
        if service is None:
            return ""
        evaluator_cfg = getattr(self.cfg, "evaluator", None)
        if getattr(evaluator_cfg, "enabled", True) is False:
            return ""
        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                cfg=self.cfg,
            )
            request = EvaluationRequest(context=ctx, trigger="prompt")
            return service.build_prompt_contract(request)
        except Exception:
            logger.debug("[lnr] evaluator prompt contract failed", exc_info=True)
            return ""

    def _record_evaluator_stage_events(
        self, *, stage_id: str, metric_event: dict[str, Any]
    ) -> dict[str, Any]:
        service = getattr(self, "assessment_pipeline", None)
        if service is None:
            service = getattr(self, "gate_service", None) or getattr(
                self, "evaluation_service", None
            )
        if service is None:
            return metric_event
        mode = self._evaluator_stage_source_mode()

        def merge_facts(facts: dict[str, Any]) -> dict[str, Any]:
            if mode == "shadow":
                return metric_event
            if mode == "adjudicate":
                return merge_adjudicated_stage_facts(metric_event, facts)
            return merge_primary_stage_facts(metric_event, facts)

        def failed_facts(reason_code: str, message: str) -> dict[str, Any]:
            return {
                "validation_ok": False,
                "candidate_ready": False,
                "selection_eligible": False,
                "metric_validity": "low",
                "metric_validity_reason_code": reason_code,
                "metric_source_note": message,
                "evaluator_backend": self._evaluator_backend_name(),
                "evaluator_status": reason_code,
                "gate_action": "retry",
                "gate_accepted": False,
                "gate_reason_code": reason_code,
                "gate_message": message,
                "_gate_evaluated": True,
            }

        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                stage_id=str(stage_id or ""),
                cfg=self.cfg,
                wall_clock_remaining_sec=max(
                    0.0, float(self.deadline - time.monotonic())
                ),
                metadata={"metric_event": dict(metric_event or {})},
            )
            request = EvaluationRequest(context=ctx, trigger="stage_end")
            assess = getattr(service, "assess", None)
            outcomes = list(
                assess(request) if callable(assess) else service.evaluate(request)
            )
            selected_facts: dict[str, Any] = {}
            for outcome in outcomes:
                event = outcome.event
                facts = metric_event_to_stage_facts(event)
                gate_trace = dict((event.extra or {}).get("gate") or {})
                facts.update(
                    {
                        # Preserve the evidence level seen by Gate separately
                        # from later Stage-result adjudication, which may lower
                        # metric_validity for selection without rewriting the
                        # historical Gate decision.
                        "gate_metric_validity": event.metric_validity,
                        "gate_policy": str(gate_trace.get("policy") or ""),
                        "gate_policy_version": str(gate_trace.get("version") or ""),
                        "gate_action": outcome.decision.action,
                        "gate_accepted": outcome.decision.accepted,
                        "gate_reason_code": outcome.decision.reason_code,
                        "gate_message": outcome.decision.message,
                        "_gate_evaluated": True,
                    }
                )
                self._jsonl(
                    self._evaluator_event_log_name(),
                    {
                        "event": "evaluator_metric_event",
                        "gate_decision": outcome.decision.to_dict(),
                        "stage_source_mode": mode,
                        "stage_facts": facts,
                        **event.to_dict(),
                    },
                )
                if not selected_facts:
                    selected_facts = facts
            if len(outcomes) != 1:
                reason = (
                    "evaluator_no_outcome"
                    if not outcomes
                    else "evaluator_multiple_outcomes"
                )
                message = (
                    "evaluator did not produce a candidate outcome"
                    if not outcomes
                    else f"evaluator produced {len(outcomes)} outcomes for one stage candidate"
                )
                self._jsonl(
                    self._evaluator_event_log_name(),
                    {
                        "event": "stage_gate_failed_closed",
                        "stage_id": stage_id,
                        "stage_source_mode": mode,
                        "reason_code": reason,
                        "message": message,
                    },
                )
                return merge_facts(failed_facts(reason, message))
            return merge_facts(selected_facts)
        except Exception as exc:
            logger.debug("[lnr] evaluator stage event failed", exc_info=True)
            reason = "evaluator_service_exception"
            message = f"{type(exc).__name__}: {exc}"
            self._jsonl(
                self._evaluator_event_log_name(),
                {
                    "event": "stage_gate_failed_closed",
                    "stage_id": stage_id,
                    "stage_source_mode": mode,
                    "reason_code": reason,
                    "message": message,
                },
            )
            return merge_facts(failed_facts(reason, message))
