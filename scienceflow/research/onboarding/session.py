"""State machine for the lightweight long-research onboarding flow.

This module deliberately stops at collecting a normal parallel-manifest input.
It does not own execution, evaluation, or resource allocation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from scienceflow.research.onboarding.support.answers import (
    detach_conversation,
    extract_task_text,
    parse_constraints,
    parse_cpu_list,
    parse_duration,
    parse_field,
    parse_gpu_selection,
    parse_workers,
)
from scienceflow.research.onboarding.support.task_defaults import (
    apply_task_package,
    may_defer_metric_direction,
    resolve_task_package,
)
from scienceflow.runtime.task_package import TaskPackageSpec


class OnboardingState(str, Enum):
    DRAFTING = "drafting"
    ASKING = "asking"
    PREFLIGHT = "preflight"
    CONFIRM = "confirm"
    RUNNING = "running"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class LongResearchDraft:
    """Short-lived answers used to produce the existing Parallel manifest."""

    task_text: str = ""
    exp_id: str = ""
    run_id: str = ""
    workspace_base: str = ""
    input_data_dir: str = ""
    metric_name: str = ""
    lower_is_better: bool | None = None
    artifact_path: str = ""
    artifact_kind: str = ""
    artifact_command: str = ""
    evaluator: dict[str, object] = field(default_factory=dict)
    gate: dict[str, object] = field(default_factory=dict)
    workers: int | None = None
    cpu_list: str = ""
    gpu_list: str = ""
    wall_clock_sec: int | None = None
    code_models: list[str] = field(default_factory=list)
    feedback_models: list[str] = field(default_factory=list)
    model_selection: str = ""
    model_config_path: str = ""
    task_profile: str = ""
    registered_task: bool = False
    exploratory: bool = False
    task_root: str = ""
    description_sources: tuple[str, ...] = ()
    discovered_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OnboardingQuestion:
    field: str
    prompt: str


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    state: OnboardingState
    action: str
    message: str
    question: OnboardingQuestion | None = None


_QUESTIONS: dict[str, str] = {
    "task_text": "What should the long research task accomplish?",
    "input_data_dir": "Where is the input dataset directory? Enter none if this task needs no external data.",
    "metric_name": "What is the primary metric name?",
    "lower_is_better": "Should the metric be minimized or maximized?",
    "artifact_path": "What relative artifact path should workers produce?",
    "artifact_command": (
        "What trusted command evaluates that artifact? "
        "This unregistered task will remain explicitly exploratory."
    ),
    "gpu_list": "GPU selection (cpu, auto, auto:N, or IDs such as 0,1)?",
    "workers": "How many parallel workers?",
    "cpu_list": "Which total CPU pool may the workers use (for example 0-31)?",
    "wall_clock_sec": (
        "Research duration (minimum 10m; for example 20m, 2h, or 7200s)?"
    ),
}

_FIELD_ORDER = (
    "task_text",
    "input_data_dir",
    "metric_name",
    "lower_is_better",
    "artifact_path",
    "artifact_command",
    "gpu_list",
    "workers",
    "cpu_list",
    "wall_clock_sec",
)


class LongResearchSession:
    """Small deterministic state machine suitable for a TUI interceptor."""

    def __init__(
        self,
        draft: LongResearchDraft,
        *,
        conversation: tuple[tuple[str, str], ...] = (),
        task_package: TaskPackageSpec | None = None,
    ) -> None:
        self.draft = draft
        self.conversation = tuple(
            (str(role), str(content)) for role, content in conversation
        )
        self.task_package = task_package
        self.state = OnboardingState.DRAFTING
        self.confirmed = False
        self.preflight_report: object | None = None
        self._refresh_state()

    @classmethod
    def start(
        cls,
        conversation: Sequence[object] = (),
        *,
        workspace: str | Path | None = None,
        constraints: str = "",
        task_id: str = "",
        natural_language: bool = False,
    ) -> LongResearchSession:
        """Detach the conversation and initialize only facts already available."""

        detached = detach_conversation(conversation)
        task_text = extract_task_text(detached)
        root = Path(workspace or Path.cwd()).expanduser().resolve(strict=False)
        draft = LongResearchDraft(
            task_text=task_text,
            workspace_base=str(root / "runs"),
        )
        values = ({"task_text": constraints.strip()} if constraints.strip() else {}) if natural_language else parse_constraints(constraints)
        if task_id.strip():
            values["exp_id"] = task_id.strip()
        _apply_values(draft, values)
        if draft.input_data_dir and draft.input_data_dir != "none":
            data = Path(draft.input_data_dir).expanduser()
            draft.input_data_dir = str((data if data.is_absolute() else root / data).resolve())

        from scienceflow.research.onboarding.support.preparation.discovery import (
            discover_task,
        )

        # Resolve registered IDs before reading arbitrary local task documents.
        package = resolve_task_package(draft.exp_id, draft.task_text, workspace=root)
        if package is not None:
            apply_task_package(draft, package)
        else:
            if natural_language:
                draft.task_root = str(root)
            else:
                discover_task(draft, root)
            draft.exploratory = True
        return cls(
            draft,
            conversation=detached,
            task_package=package,
        )

    @property
    def ready_for_preflight(self) -> bool:
        return self.next_question() is None and self.state not in {
            OnboardingState.CANCELLED,
            OnboardingState.RUNNING,
        }

    def next_question(self) -> OnboardingQuestion | None:
        if self.state in {
            OnboardingState.PREFLIGHT,
            OnboardingState.CONFIRM,
            OnboardingState.RUNNING,
            OnboardingState.CANCELLED,
        }:
            return None
        for name in _FIELD_ORDER:
            if self._is_missing(name):
                return OnboardingQuestion(name, _QUESTIONS[name])
        return None

    def next_prompt(self) -> str:
        question = self.next_question()
        if question is not None:
            return question.prompt
        if self.state == OnboardingState.PREFLIGHT:
            return "Configuration is complete; checking before launch."
        if self.state == OnboardingState.CONFIRM:
            label = "exploratory" if self.draft.exploratory else "verified"
            return f"Preflight status is {label}. Type run to start or cancel to return to chat."
        if self.state == OnboardingState.RUNNING:
            return "Long research is running."
        return "Long research onboarding was cancelled."

    def answer(self, text: str) -> SessionUpdate:
        raw = str(text or "").strip()
        command = raw.casefold()
        if command in {"cancel", "/cancel", "取消"}:
            return self.cancel()
        if self.state == OnboardingState.CANCELLED:
            return self._update("ignored", "Onboarding is already cancelled.")
        if self.state == OnboardingState.RUNNING:
            return self._update("ignored", "Long research is already running.")
        if command in {"confirm", "/confirm", "yes", "确认"}:
            return self.confirm()
        if command in {"run", "/run", "运行"}:
            return self.run()
        if self.state in {OnboardingState.PREFLIGHT, OnboardingState.CONFIRM}:
            return self._update("awaiting_action", self.next_prompt())

        question = self.next_question()
        if question is None:
            self._refresh_state()
            return self._update("preflight", self.next_prompt())
        try:
            value = parse_field(question.field, raw)
        except ValueError as exc:
            return self._update("invalid", str(exc), question=question)
        setattr(self.draft, question.field, value)
        if question.field == "task_text" and self.task_package is None:
            package = resolve_task_package(self.draft.exp_id, self.draft.task_text)
            if package is not None:
                self.task_package = package
                apply_task_package(self.draft, package)
        self._refresh_state()
        next_question = self.next_question()
        if self.state == OnboardingState.PREFLIGHT:
            return self._update("preflight", self.next_prompt())
        return self._update("answered", next_question.prompt, question=next_question)

    def revise(self, values: Mapping[str, object]) -> SessionUpdate:
        """Validate an edit atomically and require a fresh preflight afterward."""
        if self.state in {OnboardingState.RUNNING, OnboardingState.CANCELLED}:
            return self._update("ignored", self.next_prompt())
        parsed = {name: parse_field(name, value) for name, value in values.items()}
        _apply_values(self.draft, parsed)
        self.confirmed = False
        self.preflight_report = None
        self.state = OnboardingState.ASKING
        self._refresh_state()
        return self._update("preflight" if self.ready_for_preflight else "answered", self.next_prompt(),
                            question=self.next_question())

    def apply_preflight(self, report: object) -> SessionUpdate:
        """Record a report exposing ``status`` and advance only non-failed runs."""

        if self.state != OnboardingState.PREFLIGHT:
            raise RuntimeError(
                "preflight can only be applied after all questions are answered"
            )
        raw_status = getattr(report, "status", "failed") or "failed"
        status = str(getattr(raw_status, "value", raw_status)).casefold()
        if status == "failed":
            self.preflight_report = report
            return self._update(
                "preflight_failed",
                "Preflight failed; fix the reported checks or cancel.",
            )
        self.preflight_report = report
        self.draft.exploratory = status == "exploratory"
        self.state = OnboardingState.CONFIRM
        return self._update("confirm", self.next_prompt())

    def confirm(self) -> SessionUpdate:
        if self.state != OnboardingState.CONFIRM:
            return self._update("not_ready", self.next_prompt())
        self.confirmed = True
        return self._update(
            "confirmed", "Confirmed. Type run to start or cancel to return to chat."
        )

    def run(self) -> SessionUpdate:
        if self.state != OnboardingState.CONFIRM:
            return self._update("not_ready", self.next_prompt())
        self.confirmed = True
        self.state = OnboardingState.RUNNING
        return self._update("run", "Starting long research.")

    def cancel(self) -> SessionUpdate:
        self.state = OnboardingState.CANCELLED
        return self._update(
            "cancel", "Long research onboarding cancelled; returning to chat."
        )

    def _refresh_state(self) -> None:
        if self.state in {
            OnboardingState.CANCELLED,
            OnboardingState.RUNNING,
            OnboardingState.CONFIRM,
        }:
            return
        self.state = (
            OnboardingState.ASKING
            if any(self._is_missing(name) for name in _FIELD_ORDER)
            else OnboardingState.PREFLIGHT
        )

    def _is_missing(self, name: str) -> bool:
        if name == "artifact_command" and self.draft.registered_task:
            return False
        value = getattr(self.draft, name)
        if name == "lower_is_better":
            return value is None and not may_defer_metric_direction(self.draft)
        if name in {"workers", "wall_clock_sec"}:
            return value is None
        return not bool(str(value or "").strip())

    def _update(
        self,
        action: str,
        message: str,
        *,
        question: OnboardingQuestion | None = None,
    ) -> SessionUpdate:
        return SessionUpdate(self.state, action, message, question)


def _apply_values(draft: LongResearchDraft, values: Mapping[str, object]) -> None:
    for name, value in values.items():
        setattr(draft, name, value)


__all__ = [
    "LongResearchDraft",
    "LongResearchSession",
    "OnboardingQuestion",
    "OnboardingState",
    "SessionUpdate",
    "detach_conversation",
    "extract_task_text",
    "parse_constraints",
    "parse_cpu_list",
    "parse_duration",
    "parse_gpu_selection",
    "parse_workers",
]
