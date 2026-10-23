"""Worker reduction and final artifact production."""

from scienceflow.research.quality.finalization.selection.ranking import metric_float
from scienceflow.research.quality.finalization.artifacts.materialize import materialize_best_stage_final
from scienceflow.research.quality.finalization.service import run_global_merge
from scienceflow.research.quality.finalization.service import FinalizationService
from scienceflow.research.quality.finalization.worker_outcomes import (
    multi_worker_failure_kind,
    multi_worker_stop_reason,
    worker_error_kind,
)

__all__ = [
    "materialize_best_stage_final",
    "FinalizationService",
    "metric_float",
    "multi_worker_failure_kind",
    "multi_worker_stop_reason",
    "run_global_merge",
    "worker_error_kind",
]
