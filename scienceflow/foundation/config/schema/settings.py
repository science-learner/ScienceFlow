"""Stable configuration API assembled from responsibility modules."""

# This module is an explicit compatibility facade; every import is a re-export.
# ruff: noqa: F401

from scienceflow.foundation.config.schema.models.contracts.environment import _apply_env, _stage_endpoint_pairs
from scienceflow.foundation.config.schema.models.loading.parallel_overrides import (
    _apply_parallel_manifest_lnr_payload,
    apply_parallel_manifest_agent_overrides,
    apply_parallel_manifest_lnr_overrides,
)
from scienceflow.foundation.config.schema.models.contracts.profiles import (
    apply_profile_overrides,
    apply_repl_manifest_defaults,
    merge_parallel_manifest_cfg_patch,
)
from scienceflow.foundation.config.schema.models.contracts.schema import (
    AgentConfig,
    Config,
    EvaluatorCandidateConfig,
    EvaluatorCommandConfig,
    EvaluatorConfig,
    EvaluatorMetricConfig,
    ExecConfig,
    GateConfig,
    InitWorkspaceConfig,
    LnrConfig,
    ReplConfig,
    StageConfig,
    ToolConfig,
    WorkspaceConfig,
)
from scienceflow.foundation.config.schema.models.loading.serialization import (
    dump_resolved_config_yaml,
    load_cfg,
    prep_cfg,
)
from scienceflow.foundation.config.runtime.resource_modes import expand_lnr_resource_control_mode_payload

__all__ = tuple(name for name in globals() if not name.startswith("__"))
