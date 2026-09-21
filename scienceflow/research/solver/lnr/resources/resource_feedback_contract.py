"""Compatibility facade for the Resource Management feedback contract."""

from scienceflow.research.control.resources.runtime.feedback import (
    eta_bucket,
    normalize_eta_confidence,
    resource_feedback_text,
)

__all__ = ["eta_bucket", "normalize_eta_confidence", "resource_feedback_text"]
