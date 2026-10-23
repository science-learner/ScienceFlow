"""Compatibility facade for the top-level Admission policy."""

from scienceflow.research.control.admission.policy import (
    admission_feedback_from_result,
    apply_admission_decision,
    build_resource_admission_prompt,
    normalize_admission_decision,
    parse_admission_decision_text,
    should_request_admission_llm,
)

__all__ = [
    "admission_feedback_from_result",
    "apply_admission_decision",
    "build_resource_admission_prompt",
    "normalize_admission_decision",
    "parse_admission_decision_text",
    "should_request_admission_llm",
]
