# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: resource classification, cadence facts, observation feedback, and safety gates.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    Path,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_LIGHT_GPU_PROBE,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceJob,
    ResourceSourceHint,
    TRAIN_CLASSES,
    build_research_cadence_facts,
    hashlib,
    is_comparable_live_metric,
    normalize_execution_scale,
    normalize_route_key,
    re,
    resource_feedback_text,
    route_metric_evidence,
)


class ObservationCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    @staticmethod
    def _digest(command: str) -> str:
        return hashlib.sha256(str(command or "").encode("utf-8", errors="replace")).hexdigest()[:16]


    @staticmethod
    def _has_gpu_intent(source_hint: ResourceSourceHint | None) -> bool:
        if bool(getattr(source_hint, "command_cpu_only", False)):
            return False
        return bool(source_hint and source_hint.has_gpu_evidence)


    @staticmethod
    def _clip_score(value: float) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score != score:
            return 0.0
        return max(0.0, min(1.0, score))


    @staticmethod
    def _command_entrypoint(command: str) -> str:
        text = str(command or "")
        match = re.search(r"\b(?:python|python3|python3\.\d+)\s+([^\s;&|]+\.py)", text)
        if match:
            return match.group(1)[:240]
        match = re.search(r"(?:open|Path)\(\s*['\"]?([^'\"\)\s]+\.py)['\"]?", text)
        if match:
            return match.group(1)[:240]
        return ""


    def _research_route_key_for_job(self, job: ResourceJob) -> str:
        explicit = normalize_route_key((job.value_hint or {}).get("route_id"))
        if explicit:
            return explicit
        entrypoint = self._command_entrypoint(job.command)
        workspace = job.workspace_dir
        if not entrypoint or workspace is None:
            return ""
        root = Path(workspace).resolve(strict=False)
        path = (root / entrypoint).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            return ""
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""
        return f"source:{digest[:16]}"


    def _research_route_metric_evidence(self, route_key: str) -> dict[str, Any]:
        route = normalize_route_key(route_key)
        evidence = dict(self._research_route_metrics.get(route) or {})
        if self.task_resource_dir is not None and route:
            persisted = route_metric_evidence(
                self.task_resource_dir.parent / "lhr_stage_performance.csv",
                route,
            )
            if int(persisted.get("comparable_metric_count") or 0) >= int(evidence.get("comparable_metric_count") or 0):
                evidence = persisted
        return {
            "route_key": route,
            "comparable_metric_count": int(evidence.get("comparable_metric_count") or 0),
            "route_metric_proven": bool(evidence.get("route_metric_proven")),
            "latest_stage_id": str(evidence.get("latest_stage_id") or ""),
            "latest_metric_value": evidence.get("latest_metric_value"),
            "source": str(evidence.get("source") or "none"),
        }


    def _update_research_route_metric(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        route_key = self._research_route_key_for_job(job)
        metric = self._current_structured_metric_for_job(job)
        if not route_key or not is_comparable_live_metric(metric, saw_final_score=bool(signal.get("saw_final_score"))):
            return
        previous = dict(self._research_route_metrics.get(route_key) or {})
        self._research_route_metrics[route_key] = {
            "route_key": route_key,
            "comparable_metric_count": max(1, int(previous.get("comparable_metric_count") or 0)),
            "route_metric_proven": True,
            "latest_stage_id": str(previous.get("latest_stage_id") or ""),
            "latest_metric_value": metric.get("value") if isinstance(metric, dict) else None,
            "source": "live_comparable_metric",
        }


    @staticmethod
    def _research_cadence_eligible(job: ResourceJob) -> bool:
        if str(job.resource_class or "") in {
            RESOURCE_HEAVY_CPU_CANDIDATE,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }:
            return True
        hint = job.source_hint or ResourceSourceHint()
        return bool(hint.has_train_evidence or hint.has_feature_evidence or hint.has_tt_evidence)


    def _research_cadence_fact_card(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        progress_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        route_key = self._research_route_key_for_job(job)
        evidence = self._research_route_metric_evidence(route_key)
        metric = self._current_structured_metric_for_job(job)
        return build_research_cadence_facts(
            enabled=self.research_cadence_enabled,
            eligible=self._research_cadence_eligible(job),
            elapsed_sec=float(progress_snapshot.get("runtime_sec") or signal.get("elapsed_sec") or 0.0),
            observe_sec=self.research_cadence_observe_sec,
            first_metric_budget_sec=self.first_comparable_metric_budget_sec,
            proven_route_metric_budget_sec=self.proven_route_metric_budget_sec,
            execution_scale=normalize_execution_scale((job.value_hint or {}).get("execution_scale")),
            route_key=route_key,
            route_evidence=evidence,
            current_comparable_metric=is_comparable_live_metric(
                metric,
                saw_final_score=bool(signal.get("saw_final_score")),
            ),
            eta_to_next_comparable_metric_sec=self._float_or_none(
                progress_snapshot.get("eta_to_next_comparable_metric_sec")
            ),
            eta_confidence=str(progress_snapshot.get("eta_confidence") or "low"),
        )


    def _infer_value_hint(
        self,
        *,
        command: str,
        resource_class: str,
        source_hint: ResourceSourceHint | None,
        timeout_sec: float,
    ) -> dict[str, Any]:
        cmd = str(command or "").lower()
        hint = source_hint or ResourceSourceHint()
        cls = str(resource_class or "")
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN} or hint.has_train_evidence:
            stage_type = "train"
            expected = 0.55
        elif cls == RESOURCE_GPU_FEATURE_EXTRACT or hint.has_feature_evidence:
            stage_type = "feature_extract"
            expected = 0.45
        elif cls == RESOURCE_GPU_TT_LIGHT or hint.has_tt_evidence:
            stage_type = "submission_or_inference"
            expected = 0.65
        else:
            stage_type = "unknown_gpu"
            expected = 0.35
        near_submission = 0.0
        if re.search(r"\b(submission|submit|predict|inference|infer|tta|ensemble|blend)\b", cmd):
            near_submission = 0.85
            expected = max(expected, 0.65)
        elif re.search(r"\b(valid|validation|evaluate|eval|score)\b", cmd):
            near_submission = 0.45
        elif stage_type == "train":
            near_submission = 0.20
        requested = hint.requested_gpu_count or 0
        diversity = 0.2 if requested <= 1 else 0.1
        timeout = max(0.0, float(timeout_sec or 0.0))
        long_runtime = self._clip_score(timeout / 3600.0) if timeout > 0 else (0.5 if stage_type == "train" else 0.2)
        return {
            "stage_type": stage_type,
            "expected_value_score": self._clip_score(expected),
            "lineage_diversity_score": diversity,
            "near_submission_score": self._clip_score(near_submission),
            "worker_starvation_score": 0.0,
            "duplicate_penalty": 0.0,
            "timeout_history_penalty": 0.0,
            "long_runtime_penalty": self._clip_score(long_runtime),
            "requested_gpu_count": requested,
        }


    def _refine_resource_class(
        self,
        resource_class: str,
        *,
        source_hint: ResourceSourceHint | None,
    ) -> str:
        cls = str(resource_class or "")
        hint = source_hint or ResourceSourceHint()
        if bool(getattr(hint, "command_cpu_only", False)):
            return self._cpu_only_resource_class(cls, hint)
        has_gpu = self._has_gpu_intent(hint)
        if cls == RESOURCE_UNKNOWN_EXEC:
            return self._unknown_resource_class(hint, has_gpu=has_gpu)
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            return self._gpu_evidence_resource_class(hint, fallback=cls)
        if cls in {RESOURCE_LIGHT_CPU, RESOURCE_LIGHT_GPU_PROBE, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_PURE_TT_CPU} and has_gpu:
            return self._gpu_evidence_resource_class(hint, fallback=RESOURCE_UNKNOWN_GPU_EXEC if getattr(hint, "command_gpu_compute_evidence", False) else cls)
        if cls == RESOURCE_HEAVY_GPU_CANDIDATE and has_gpu and not hint.has_train_evidence:
            return self._gpu_evidence_resource_class(hint, fallback=cls, include_train=False)
        if cls == RESOURCE_GPU_TT_LIGHT and not has_gpu:
            return RESOURCE_PURE_TT_CPU
        if cls == RESOURCE_PURE_TT_CPU and has_gpu:
            return RESOURCE_GPU_TT_LIGHT
        return cls

    @staticmethod
    def _cpu_only_resource_class(resource_class: str, hint: ResourceSourceHint) -> str:
        if resource_class in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN}:
            return RESOURCE_HEAVY_CPU_CANDIDATE
        if resource_class == RESOURCE_UNKNOWN_GPU_EXEC:
            return RESOURCE_HEAVY_CPU_CANDIDATE if (hint.has_train_evidence or hint.has_feature_evidence) else RESOURCE_UNKNOWN_EXEC
        if resource_class == RESOURCE_GPU_TT_LIGHT:
            return RESOURCE_PURE_TT_CPU
        if resource_class == RESOURCE_LIGHT_GPU_PROBE:
            return RESOURCE_LIGHT_CPU
        return resource_class

    @staticmethod
    def _gpu_evidence_resource_class(hint: ResourceSourceHint, *, fallback: str, include_train: bool=True) -> str:
        if include_train and hint.has_train_evidence:
            return RESOURCE_HEAVY_GPU_CANDIDATE
        if hint.has_feature_evidence:
            return RESOURCE_GPU_FEATURE_EXTRACT
        if hint.has_tt_evidence:
            return RESOURCE_GPU_TT_LIGHT
        return fallback

    def _unknown_resource_class(self, hint: ResourceSourceHint, *, has_gpu: bool) -> str:
        if not has_gpu:
            return RESOURCE_UNKNOWN_EXEC
        return self._gpu_evidence_resource_class(hint, fallback=RESOURCE_UNKNOWN_GPU_EXEC)


    def _merge_resource_class(self, current: str, incoming: str, *, source_hint: ResourceSourceHint | None) -> str:
        current_cls = self._refine_resource_class(current, source_hint=source_hint)
        incoming_cls = self._refine_resource_class(incoming, source_hint=source_hint)
        if not incoming_cls:
            return current_cls
        if current_cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_PURE_TT_CPU} and incoming_cls in TRAIN_CLASSES:
            hint = source_hint or ResourceSourceHint()
            return incoming_cls if hint.has_train_evidence else current_cls
        if current_cls == RESOURCE_UNKNOWN_EXEC:
            return incoming_cls
        if current_cls == RESOURCE_UNKNOWN_GPU_EXEC and incoming_cls != RESOURCE_UNKNOWN_EXEC:
            return incoming_cls
        return current_cls or incoming_cls


    def _gpu_queue_relevant(self, resource_class: str, source_hint: ResourceSourceHint | None, gpu_ids: list[str]) -> bool:
        if self.resource_runtime is None:
            return False
        cls = str(resource_class or "")
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        queue_classes = {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }
        if ids and cls in queue_classes:
            if self._has_gpu_intent(source_hint):
                return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)
            hint = source_hint or ResourceSourceHint()
            has_entrypoint = bool(hint.entrypoints)
            has_stage_hint = bool(hint.has_train_evidence or hint.has_feature_evidence or hint.has_tt_evidence)
            source_checked = int(getattr(hint, "source_files_inspected", 0) or 0) > 0
            if has_entrypoint and has_stage_hint and (not source_checked or hint.has_feature_evidence or hint.has_tt_evidence):
                return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)
            return False
        if not self._has_gpu_intent(source_hint):
            return False
        return self.resource_runtime.should_queue_gpu(resource_class=cls, gpu_ids=ids)


    def _emit(self, event: str, job: ResourceJob, *, status: str = "", payload: dict[str, Any] | None = None) -> None:
        body = {
            "job_id": job.job_id,
            "command_digest": job.command_digest,
            "resource_class": job.resource_class,
            "gpu_ids": job.gpu_ids,
            "cpu_set": job.cpu_set,
            "timeout_sec": job.timeout_sec,
            **(payload or {}),
        }
        self.state_machine.append_event(
            event,
            task_type="resource_job",
            task_id=f"resource_job:{job.job_id}",
            status=status,
            payload=body,
        )


    def _resource_feedback_text(
        self,
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
        holder_job_id: str = "",
        queue_position: int | None = None,
        queue_len: int | None = None,
        cooldown_sec: float | None = None,
        eta_next_train_sec: float | None = None,
        eta_confidence: str = "",
        pressure_generation: int | None = None,
        duplicate_digest_count: int | None = None,
        unlock_condition: str = "",
        blocked_until_unlock: bool | None = None,
        schema_state: str = "",
        artifact_state: str = "",
        progress_state: str = "",
        extra_facts: dict[str, Any] | None = None,
    ) -> str:
        return resource_feedback_text(
            status=status,
            reason=reason,
            scope=scope,
            resource_mode=resource_mode,
            blocked_class=blocked_class,
            gpu_ids=gpu_ids,
            allowed_classes=allowed_classes,
            holder_job_id=holder_job_id,
            queue_position=queue_position,
            queue_len=queue_len,
            cooldown_sec=cooldown_sec,
            eta_next_train_sec=eta_next_train_sec,
            eta_confidence=eta_confidence,
            pressure_generation=pressure_generation,
            duplicate_digest_count=duplicate_digest_count,
            unlock_condition=unlock_condition,
            blocked_until_unlock=blocked_until_unlock,
            schema_state=schema_state,
            artifact_state=artifact_state,
            progress_state=progress_state,
            extra_facts=extra_facts,
        )


    @staticmethod
    def _resource_class_token(value: Any) -> str:
        return re.sub(r"[^a-z0-9_.:-]+", "_", str(value or "").strip().lower()).strip("_")


    @staticmethod
    def _safe_candidate_artifact(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            return ""
        return path.as_posix()


    @classmethod
    def _resource_class_in_allowed(cls, resource_class: str, allowed_classes: list[str] | None) -> bool:
        blocked = cls._resource_class_token(resource_class)
        if not blocked:
            return False
        allowed = {cls._resource_class_token(value) for value in (allowed_classes or []) if str(value).strip()}
        return blocked in allowed


    @staticmethod
    def _hard_safety_gate_reason(reason: str) -> bool:
        return str(reason or "").strip().lower() in {
            "duplicate_digest_cooldown",
            "invalid_deliverable_schema_preflight",
            "task_gpu_boundary_preflight_violation",
            "workspace_gpu_boundary_violation",
            "boundary_violation",
        }


    def _allowed_class_bypass_result(
        self,
        job: ResourceJob,
        *,
        reason: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None,
    ) -> dict[str, Any] | None:
        if self._hard_safety_gate_reason(reason):
            return None
        if not self._resource_class_in_allowed(job.resource_class, allowed_classes):
            return None
        allowed = [str(x) for x in (allowed_classes or []) if str(x).strip()]
        self._emit(
            "resource_allowed_class_bypass",
            job,
            status="allowed",
            payload={
                "reason": "allowed_class_bypass",
                "bypassed_reason": str(reason or ""),
                "scope": str(scope or ""),
                "resource_mode": str(resource_mode or ""),
                "resource_class": job.resource_class,
                "allowed_classes": allowed,
            },
        )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "reason": "allowed_class_bypass",
            "bypassed_reason": str(reason or ""),
            "allowed_classes": allowed,
        }


    def _soft_gate_admission_enabled(self, *, reason: str, source_reason: str = "") -> bool:
        if self._hard_safety_gate_reason(reason) or self._hard_safety_gate_reason(source_reason):
            return False
        # Monitor-first startup owns soft resource blocks. Keep the older
        # startup-time admission review only for conservative/backward-compatible
        # runs so soft gates do not become a second pre-run decision layer.
        if self._startup_trial_enabled():
            return False
        return bool(
            self.admission_llm_enabled
            and self.admission_llm_mode != "off"
            and self.admission_decider is not None
        )


    def _soft_gate_admission_review_result(
        self,
        job: ResourceJob,
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None = None,
        feedback: str = "",
        feedback_state: dict[str, Any] | None = None,
        eta_next_train_sec: float | None = None,
        eta_confidence: str = "low",
        cooldown_sec: float | None = None,
        retry_after_sec: float | None = None,
        unlock_condition: str = "resource_context_changed",
        blocked_until_unlock: bool = True,
        extra_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gpu_ids = [str(x) for x in (job.gpu_ids or []) if str(x).strip()]
        allowed = [str(x) for x in (allowed_classes or []) if str(x).strip()]
        soft_gate = {
            "reason": str(reason or "resource_policy_gate"),
            "scope": str(scope or "task"),
            "resource_mode": str(resource_mode or "YELLOW"),
            "allowed_classes": allowed,
            "cooldown_sec": cooldown_sec,
            "retry_after_sec": retry_after_sec,
            "unlock_condition": str(unlock_condition or ""),
            "blocked_until_unlock": bool(blocked_until_unlock),
        }
        if isinstance(extra_facts, dict):
            soft_gate.update(extra_facts)
        result: dict[str, Any] = {
            "allowed": False,
            "enabled": True,
            "acquired": False,
            "requires_admission_review": True,
            "soft_gate_reason": str(reason or "resource_policy_gate"),
            "soft_gate_scope": str(scope or "task"),
            "soft_gate": soft_gate,
            "resource_class": job.resource_class,
            "policy_resource_class": job.resource_class,
            "status": str(status or "DENIED_REPLAN"),
            "admission_action": "PENDING",
            "reason": str(reason or "resource_policy_gate"),
            "resource_mode": str(resource_mode or "YELLOW"),
            "gpu_ids": gpu_ids,
            "allowed_classes": allowed,
            "requested_gpu_count": job.gpu_request_count,
            "eta_next_train_sec": eta_next_train_sec,
            "eta_confidence": str(eta_confidence or "low"),
            "retry_after_sec": retry_after_sec,
            "blocked_until_unlock": bool(blocked_until_unlock),
            "unlock_condition": str(unlock_condition or ""),
            "feedback": str(feedback or ""),
            "details": {"soft_gate": soft_gate},
        }
        if isinstance(feedback_state, dict):
            result.update({
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            })
        if self.resource_runtime is not None and job.gpu_queue_relevant:
            opportunity = self.resource_runtime.admission_opportunity_facts(
                resource_class=job.resource_class,
                gpu_ids=gpu_ids,
                request_count=job.gpu_request_count,
            )
            result.update({
                "admission_opportunity": opportunity,
                "lease_grantable_by_llm": bool(opportunity.get("lease_grantable_by_llm")),
                "candidate_physical_gpus": gpu_ids,
                "allowed_physical_gpus": gpu_ids,
                "assigned_physical_gpus": [],
            })
        return result
