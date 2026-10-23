# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Progress, artifact freshness, recoverability, and finish-feasibility facts."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (Any, Path, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_GPU_TT_LIGHT, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_LIGHT_CPU, RESOURCE_PURE_TT_CPU, RESOURCE_UNKNOWN_EXEC, RESOURCE_UNKNOWN_GPU_EXEC, ResourceJob)


class ProgressObservation:
    """Own ProgressObservation resource behavior without delegated forwarding."""

    @staticmethod
    def _elapsed_since_progress(payload: dict[str, Any], *, elapsed_sec: float) -> float:
        if not isinstance(payload, dict) or not payload:
            return max(0.0, float(elapsed_sec or 0.0))
        try:
            last_elapsed = float(payload.get("elapsed_sec") or 0.0)
        except (TypeError, ValueError):
            last_elapsed = 0.0
        if last_elapsed <= 0.0:
            return max(0.0, float(elapsed_sec or 0.0))
        return max(0.0, float(elapsed_sec or 0.0) - last_elapsed)


    @staticmethod
    def _artifact_update_is_log_like(artifact: dict[str, Any]) -> bool:
        path = str((artifact or {}).get("path") or (artifact or {}).get("name") or "").strip().lower()
        if not path:
            return False
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".log"):
            return True
        return bool(name.endswith(".txt") and any(token in name for token in ("log", "stdout", "stderr", "trace")))


    @classmethod
    def _artifact_update_is_value_bearing(
        cls,
        artifact: dict[str, Any],
        *,
        deliverable_validity: str = "none",
    ) -> bool:
        if not isinstance(artifact, dict) or cls._artifact_update_is_log_like(artifact):
            return False
        path = str(artifact.get("path") or artifact.get("name") or "").strip().lower()
        if not path:
            return False
        try:
            if int(artifact.get("size_bytes") or 0) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        stability = str(artifact.get("stability") or "").strip().lower()
        if stability not in {"stable", "done_marker", "run_state_confirmed"}:
            return False
        scope = str(artifact.get("artifact_scope") or "").strip().lower()
        if scope not in {"current_run", "workspace_recent"}:
            return False
        name = path.rsplit("/", 1)[-1]
        if name in {"submission.csv", "predictions.csv"}:
            return str(deliverable_validity or "").strip().lower() == "produced_valid"
        suffix = Path(name).suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".joblib", ".npy", ".npz"}:
            return True
        if suffix == ".csv" and any(
            token in name for token in ("submission", "submit", "pred", "prediction", "oof", "logit", "prob")
        ):
            return True
        return False


    def _fresh_artifact_update_for_job(self, job: ResourceJob, *, elapsed_sec: float) -> dict[str, Any]:
        payload = job.last_artifact_progress if isinstance(job.last_artifact_progress, dict) else {}
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        artifact = signals.get("artifact") if isinstance(signals.get("artifact"), dict) else {}
        if not artifact:
            return {}
        out = dict(artifact)
        payload_elapsed = self._float_or_none(payload.get("elapsed_sec"))
        base_age = self._float_or_none(out.get("age_sec"))
        if payload_elapsed is not None and payload_elapsed > 0.0:
            delta = max(0.0, float(elapsed_sec or 0.0) - payload_elapsed)
            out["age_sec"] = max(0.0, float(base_age or 0.0) + delta)
            out["observed_at_elapsed_sec"] = payload_elapsed
        else:
            out["age_sec"] = self._elapsed_since_progress(payload, elapsed_sec=elapsed_sec)
        out["age_refreshed_at_elapsed_sec"] = max(0.0, float(elapsed_sec or 0.0))
        out["artifact_log_like"] = self._artifact_update_is_log_like(out)
        return out


    def _fresh_recoverability_for_job(self, job: ResourceJob, *, elapsed_sec: float) -> dict[str, Any]:
        rec = dict(job.last_recoverability or {}) if isinstance(job.last_recoverability, dict) else {}
        if not rec:
            return {}
        observed = self._float_or_none(rec.get("observed_at_elapsed_sec"))
        if observed is None or observed <= 0.0:
            payload = job.last_artifact_progress if isinstance(job.last_artifact_progress, dict) else {}
            observed = self._float_or_none(payload.get("elapsed_sec"))
        delta = max(0.0, float(elapsed_sec or 0.0) - float(observed or elapsed_sec or 0.0))
        for key in ("artifact_age_sec", "last_recoverable_artifact_age_sec"):
            if rec.get(key) is None:
                continue
            age = self._float_or_none(rec.get(key))
            if age is not None:
                rec[key] = max(0.0, age + delta)
        if rec.get("artifact_age_sec") is None:
            artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
            if artifact:
                rec["artifact_age_sec"] = artifact.get("age_sec")
        rec["age_refreshed_at_elapsed_sec"] = max(0.0, float(elapsed_sec or 0.0))
        return rec


    def _stdout_observation_for_job(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        stdout_age = self._float_or_none(signal.get("stdout_age_sec"))
        stdout_lines = int(signal.get("stdout_lines") or 0)
        stdout_bytes = int(signal.get("stdout_bytes") or 0)
        stream_present = bool(stdout_lines > 0 or stdout_bytes > 0)
        stream_limit = max(0.0, float(self.stalled_stdout_sec or 0.0))
        stream_fresh = bool(stream_present and (stdout_age is None or stream_limit <= 0.0 or stdout_age < stream_limit))
        artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        redirected_present = bool(artifact and self._artifact_update_is_log_like(artifact))
        redirected_age = self._float_or_none(artifact.get("age_sec")) if artifact else None
        redirected_limit = max(0.0, float(self.low_progress_no_artifact_sec or self.stalled_stdout_sec or 300.0))
        if redirected_limit <= 0.0:
            redirected_limit = 300.0
        redirected_fresh = bool(redirected_present and redirected_age is not None and redirected_age < redirected_limit)
        if stream_present:
            primary = "stdout_stream"
        elif redirected_present:
            primary = "redirected_log"
        else:
            primary = "none"
        return {
            "primary_channel": primary,
            "stdout_stream": {
                "present": stream_present,
                "fresh": stream_fresh,
                "age_sec": stdout_age,
                "lines": stdout_lines,
                "bytes": stdout_bytes,
            },
            "redirected_log": {
                "present": redirected_present,
                "fresh": redirected_fresh,
                "age_sec": redirected_age,
                "path": str(artifact.get("path") or "") if artifact else "",
                "size_bytes": int(artifact.get("size_bytes") or 0) if artifact else 0,
            },
            "stalled": bool((stream_present and not stream_fresh) or (redirected_present and not redirected_fresh)),
        }


    def _job_uses_expensive_gpu(self, job: ResourceJob) -> bool:
        if bool(getattr(job.source_hint, "command_cpu_only", False)):
            return False
        cls = str(job.resource_class or "")
        if cls in {RESOURCE_PURE_TT_CPU, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_UNKNOWN_EXEC, RESOURCE_LIGHT_CPU}:
            return False
        if self.resource_runtime is not None and self.resource_runtime.has_active_lease(job_id=job.job_id):
            return True
        if job.gpu_queue_relevant:
            return True
        if cls in {RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_GPU_TT_LIGHT, RESOURCE_UNKNOWN_GPU_EXEC}:
            return bool(job.gpu_ids) or self._has_gpu_intent(job.source_hint)
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return bool(job.gpu_ids) or self._has_gpu_intent(job.source_hint)
        return False


    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out else None


    def _progress_finish_feasibility(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        phase = str(signal.get("current_phase") or payload.get("phase") or signals.get("phase") or "").strip().lower()
        best: dict[str, Any] = {}
        heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
        heartbeat_elapsed = self._float_or_none(heartbeat.get("elapsed_s"))
        payload_elapsed = self._float_or_none(payload.get("elapsed_sec"))
        elapsed = max(0.0, float(elapsed_sec or 0.0), float(payload_elapsed or 0.0), float(heartbeat_elapsed or 0.0))
        structured = payload.get("structured_progress") if isinstance(payload.get("structured_progress"), dict) else {}
        for unit, raw_entry in signals.items():
            if unit in {"heartbeat", "metrics", "artifact", "phase"} or not isinstance(raw_entry, dict):
                continue
            current = self._float_or_none(raw_entry.get("current"))
            total = self._float_or_none(raw_entry.get("total"))
            if current is None or total is None or total <= 0.0 or current < 0.0 or current > total:
                continue
            if current <= 0.0 or elapsed <= 0.0:
                continue
            structured_rate = self._float_or_none(structured.get("rate_units_per_sec"))
            structured_matches = bool(
                str(structured.get("unit") or "") == str(unit)
                and self._float_or_none(structured.get("total")) == total
                and self._float_or_none(structured.get("current")) == current
            )
            rate = float(structured_rate) if structured_matches and structured_rate and structured_rate > 0.0 else current / elapsed
            if rate <= 0.0:
                continue
            interval_samples = int(structured.get("interval_sample_count") or 0) if structured_matches else 0
            progress_source = str(structured.get("source") or raw_entry.get("source") or "unknown")
            evidence_trust = str(structured.get("evidence_trust") or "unknown")
            eta = max(0.0, (total - current) / rate)
            candidate = {
                "progress_unit": str(unit),
                "progress_scope": self._progress_unit_scope(unit),
                "progress_units_done": current,
                "progress_units_total": total,
                "progress_rate_units_per_sec": rate,
                "eta_to_current_phase_end_sec": eta,
                "eta_source": f"progress_interval:{unit}" if structured_matches and structured_rate else f"progress_heartbeat:{unit}",
                "eta_confidence": "high" if interval_samples >= 2 else "medium",
                "progress_interval_samples": interval_samples,
                "progress_source": progress_source,
                "progress_evidence_trust": evidence_trust,
            }
            candidate_rank = {"subphase": 1, "phase": 2, "route": 3}[str(candidate["progress_scope"])]
            best_rank = {"subphase": 1, "phase": 2, "route": 3}.get(str(best.get("progress_scope") or ""), 0)
            if not best or candidate_rank > best_rank or (
                candidate_rank == best_rank
                and eta > float(best.get("eta_to_current_phase_end_sec") or 0.0)
            ):
                best = candidate
        if not best:
            return {
                "progress_unit": "",
                "progress_units_done": None,
                "progress_units_total": None,
                "progress_rate_units_per_sec": None,
                "eta_to_current_phase_end_sec": None,
                "eta_to_next_comparable_metric_sec": None,
                "eta_to_deliverable_sec": None,
                "eta_source": "none",
                "eta_confidence": "low",
                "finish_feasible": "unknown",
                "finish_feasibility_reason": "eta_unavailable",
                "remaining_useful_budget_sec": None,
                "progress_fraction": None,
                "structured_progress_recent": False,
                "phase_completion_protected": False,
            }
        progress_fraction = float(best["progress_units_done"]) / max(float(best["progress_units_total"]), 1.0)
        progress_age = max(0.0, elapsed - float(payload_elapsed or 0.0))
        progress_recent = bool(payload_elapsed is not None and progress_age <= max(120.0, self.review_heartbeat_sec * 2.0))
        eta_phase = float(best.get("eta_to_current_phase_end_sec") or 0.0)
        eta_source = str(best.get("eta_source") or "")
        terminal_phase = any(token in phase for token in (
            "valid",
            "eval",
            "infer",
            "predict",
            "submission",
            "score",
            "final",
            "test",
        ))
        best.update({
            "progress_fraction": progress_fraction,
            "structured_progress_recent": progress_recent,
            "structured_progress_age_sec": progress_age,
            "phase_completion_protected": bool(
                progress_recent
                and (str(best.get("progress_scope") or "") != "subphase" or terminal_phase)
                and (
                    str(best.get("eta_confidence") or "") == "high"
                    or (
                        str(best.get("eta_confidence") or "") == "medium"
                        and int(best.get("progress_interval_samples") or 0) >= 1
                        and progress_fraction >= 0.9
                    )
                )
                and eta_phase <= 300.0
                and (eta_source.startswith("progress_interval:") or progress_fraction >= 0.9)
                and not bool(signal.get("deadline_event"))
            ),
            "phase_completion_protection_sec": 300.0,
        })
        remaining = self._float_or_none(signal.get("deadline_remaining_sec"))
        metric_phase = any(token in phase for token in ("train", "fit", "epoch", "valid", "eval", "score", "oof"))
        best["eta_to_next_comparable_metric_sec"] = (
            float(best.get("eta_to_current_phase_end_sec") or 0.0)
            if metric_phase and str(best.get("progress_scope") or "") != "subphase"
            else None
        )
        reserve = max(0.0, float(self._float_or_none(signal.get("finalization_reserve_sec")) or 0.0))
        if remaining is None:
            best.update({
                "eta_to_deliverable_sec": None,
                "finish_feasible": "unknown",
                "finish_feasibility_reason": "budget_unknown",
                "remaining_useful_budget_sec": None,
            })
            return best
        useful_budget = max(0.0, remaining - reserve)
        eta_phase = float(best.get("eta_to_current_phase_end_sec") or 0.0)
        if eta_phase > useful_budget:
            finish_feasible: bool | str = False
            reason = "eta_exceeds_remaining_useful_budget"
            eta_deliverable: float | None = eta_phase
        elif terminal_phase:
            finish_feasible = True
            reason = "terminal_phase_eta_within_remaining_useful_budget"
            eta_deliverable = eta_phase
        else:
            finish_feasible = "unknown"
            reason = "phase_eta_available_deliverable_eta_unknown"
            eta_deliverable = None
        best.update({
            "eta_to_deliverable_sec": eta_deliverable,
            "finish_feasible": finish_feasible,
            "finish_feasibility_reason": reason,
            "remaining_useful_budget_sec": useful_budget,
        })
        return best
