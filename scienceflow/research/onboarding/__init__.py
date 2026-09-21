"""Public API for lightweight long-research onboarding."""

from scienceflow.research.onboarding.manifest import (
    OnboardingFiles,
    build_parallel_manifest,
    write_onboarding_files,
)
from scienceflow.research.onboarding.preflight import (
    PreflightCheck,
    PreflightReport,
    PreflightStatus,
    run_preflight,
)
from scienceflow.research.onboarding.session import (
    LongResearchDraft,
    LongResearchSession,
    OnboardingQuestion,
    OnboardingState,
    SessionUpdate,
    detach_conversation,
    extract_task_text,
    parse_constraints,
    parse_cpu_list,
    parse_duration,
    parse_gpu_selection,
    parse_workers,
)

__all__ = [
    "LongResearchDraft",
    "LongResearchSession",
    "OnboardingFiles",
    "OnboardingQuestion",
    "OnboardingState",
    "PreflightCheck",
    "PreflightReport",
    "PreflightStatus",
    "SessionUpdate",
    "build_parallel_manifest",
    "detach_conversation",
    "extract_task_text",
    "parse_constraints",
    "parse_cpu_list",
    "parse_duration",
    "parse_gpu_selection",
    "parse_workers",
    "run_preflight",
    "write_onboarding_files",
]
