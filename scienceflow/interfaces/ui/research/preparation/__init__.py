"""Task preparation scope and local execution placement."""

from .location import allocate_preparation_location
from .runtime import run_preparation_task

__all__ = ["allocate_preparation_location", "run_preparation_task"]
