"""Execute declared task-package samples through the production evaluator path."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from scienceflow.foundation.contracts import EvalContext


def run_evaluator_probe(backend, context: EvalContext, package) -> tuple[bool, str] | None:
    validation = package.config.get("validation")
    if not isinstance(validation, Mapping):
        return None
    valid = _sample(package.source_dir, validation.get("valid_artifact"))
    invalid = _sample(package.source_dir, validation.get("invalid_artifact"))
    if valid is None or invalid is None:
        return False, "validation must declare existing valid_artifact and invalid_artifact files"

    with tempfile.TemporaryDirectory(prefix="scienceflow-evaluator-probe-") as raw_root:
        root = Path(raw_root)
        probe = EvalContext(
            task_profile=context.task_profile,
            task_id=context.task_id,
            task_root=root,
            workspace=root,
            worker_id="preflight",
            stage_id="preflight",
            cfg=context.cfg,
        )
        artifact = root / package.artifact_path
        artifact.parent.mkdir(parents=True)
        shutil.copy2(valid, artifact)
        valid_candidates = backend.detect_candidates(probe)
        if len(valid_candidates) != 1:
            return False, f"valid sample produced {len(valid_candidates)} candidates"
        accepted = backend.evaluate(probe, valid_candidates[0])

        shutil.copy2(invalid, artifact)
        invalid_candidates = backend.detect_candidates(probe)
        if len(invalid_candidates) != 1:
            return False, f"invalid sample produced {len(invalid_candidates)} candidates"
        rejected = backend.evaluate(probe, invalid_candidates[0])

    valid_ok = bool(
        accepted.ok
        and accepted.validation_ok
        and accepted.selection_eligible
        and accepted.metric_value is not None
    )
    invalid_ok = not rejected.ok and not rejected.selection_eligible
    if valid_ok and invalid_ok:
        return True, "production evaluator accepted the valid sample and rejected the invalid sample"
    return (
        False,
        (
            f"valid={accepted.status}/{accepted.reason_code}; "
            f"invalid={rejected.status}/{rejected.reason_code}"
        ),
    )


def _sample(root: Path, value) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    source = (root / text).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        return None
    return source


__all__ = ["run_evaluator_probe"]
