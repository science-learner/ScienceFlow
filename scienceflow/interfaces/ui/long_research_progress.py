"""Compatibility access to the long-research progress view."""

from .research import long_research_progress as _implementation
from .research.long_research_progress import LongResearchProgress

__all__ = ["LongResearchProgress"]


def __getattr__(name):
    return getattr(_implementation, name)
