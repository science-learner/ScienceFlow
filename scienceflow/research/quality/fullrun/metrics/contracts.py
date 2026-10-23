# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Validate explicit scores and LLM interpretations against execution output."""

from __future__ import annotations

import math
import re
from typing import Any

from scienceflow.runtime.safety.policy.execution_policy import (
    final_validation_score_contract_error,
    looks_like_score_source_template,
    validation_metric_value_error,
)


def _score_contract_errors(stdout: str) -> tuple[str | None, str | None]:
    """Return ``(hard_error, soft_warning)`` for bare-run score parsing.

    Missing, unparseable, or non-finite ``Final Validation Score`` is a hard gate:
    the node has no reliable metric. A parseable score that is not the last
    stdout line is only a formatting issue; forcing another full training run for
    that case is too expensive and does not improve metric trust once submission
    validation has already passed.
    """
    hard_error = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=False,
    )
    if hard_error:
        return hard_error, None
    soft_warning = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=True,
    )
    return None, soft_warning


_MISSING_FINAL_SCORE_ERROR = (
    "missing required `Final Validation Score: <finite_float>` line"
)
_INTERPRETED_METRIC_SPLITS = {"validation", "val", "holdout", "heldout", "cv", "oof"}
_INTERPRETED_METRIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_NON_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:private|public[ _-]+leaderboard|leaderboard|test(?:ing)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:val(?:idation)?|hold[ _-]?out|cv|oof)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_FINAL_OBJECTIVE_RE = re.compile(
    r"^\s*FINAL\s+(?P<name>[A-Za-z][A-Za-z0-9_.-]*)\s*=\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)


def _extract_final_objective_metric(stdout: str) -> tuple[float, str] | None:
    """Parse the explicit final-objective contract used by artifact tasks."""
    for raw_line in reversed(str(stdout or "").splitlines()):
        match = _FINAL_OBJECTIVE_RE.fullmatch(raw_line)
        if match is None or looks_like_score_source_template(raw_line):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            return value, match.group("name")[:160]
    return None


def _validate_llm_metric_interpretation(
    stdout: str,
    interpretation: Any,
) -> tuple[float | None, str, str | None]:
    """Verify that an LLM interpretation is directly grounded in stdout."""
    if not isinstance(interpretation, dict):
        return None, "", "metric interpreter returned no structured result"
    if interpretation.get("metric_found") is not True:
        return None, "", "metric interpreter did not find a final validation metric"
    if interpretation.get("is_final") is not True:
        return None, "", "interpreted metric is not explicitly final or best"
    if str(interpretation.get("confidence") or "").strip().lower() != "high":
        return None, "", "metric interpreter confidence is not high"

    split = str(interpretation.get("split") or "").strip().lower().replace("-", "")
    if split not in _INTERPRETED_METRIC_SPLITS:
        return (
            None,
            "",
            f"interpreted metric split is not validation-like: {split or 'missing'}",
        )

    evidence = str(interpretation.get("evidence_line") or "")
    if not evidence or "\n" in evidence or "\r" in evidence:
        return None, "", "metric evidence must be one non-empty stdout line"
    if evidence not in (stdout or ""):
        return None, "", "metric evidence line is not present verbatim in stdout"
    if looks_like_score_source_template(evidence):
        return None, "", "metric evidence is a source-code template, not runtime output"
    if _NON_VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence refers to test or leaderboard data"
    if re.search(
        r"(?<![A-Za-z0-9])train(?:ing)?(?![A-Za-z0-9])",
        evidence,
        re.IGNORECASE,
    ) and not _VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence is training-only"

    metric_name = str(interpretation.get("metric_name") or "").strip()
    if not metric_name:
        return None, "", "interpreted metric name is missing"
    try:
        value = float(interpretation.get("metric_value"))
    except (TypeError, ValueError):
        return None, "", "interpreted metric value is not numeric"
    value_error = validation_metric_value_error(stdout, value)
    if value_error:
        return None, "", value_error

    evidence_values: list[float] = []
    for match in _INTERPRETED_METRIC_NUMBER_RE.finditer(evidence):
        try:
            candidate = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(candidate):
            evidence_values.append(candidate)
    if not any(
        math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-12)
        for candidate in evidence_values
    ):
        return None, "", "interpreted metric value does not occur in the evidence line"
    return value, metric_name[:160], None
