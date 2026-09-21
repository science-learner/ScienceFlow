"""TUI projections over the private model registry."""

from __future__ import annotations

import shlex
from pathlib import Path

from scienceflow.foundation.config.llm.model_registry import load_model_registry
from scienceflow.research.onboarding.session import LongResearchDraft
from scienceflow.research.onboarding.support.answers import (
    parse_constraints,
    parse_field,
)


def model_config_hint(path: Path) -> str:
    """Return a compact, secret-free pointer to the model registry."""

    selected = path.expanduser().resolve(strict=False)
    try:
        shown = f"~/{selected.relative_to(Path.home())}"
    except ValueError:
        shown = str(selected)
    return f"Config · {shown} · edit, then reopen /models"


def model_config_setup_hint() -> str:
    """Return the CLI commands needed to create or locate the registry."""

    return "Model setup · scienceflow config init · locate: scienceflow config path"


def model_alias_completions(
    path: Path,
    text: str,
) -> list[tuple[str, str]] | None:
    source = str(text)
    if source.startswith("/models "):
        prefix = source.rsplit(" ", 1)[-1]
        base = source[: -len(prefix)] if prefix else source
        if prefix.startswith("policy="):
            fragment = prefix.partition("=")[2]
            return [
                (f"{base}policy={policy}", "Routing policy")
                for policy in ("auto", "spread", "fixed")
                if policy.startswith(fragment)
            ]
        try:
            registry = load_model_registry(path)
            aliases = list(registry.default_aliases("code_models"))
        except ValueError:
            return []
        selected, comma, fragment = prefix.rpartition(",")
        lead = f"{selected}," if comma else ""
        return [
            (f"{base}{lead}{alias}", "Chat model")
            for alias in aliases
            if alias.startswith(fragment) and alias not in selected.split(",")
        ]

    prefix = str(text).rsplit(" ", 1)[-1]
    key, separator, value = prefix.partition("=")
    if not separator or key not in {"models", "code-models", "feedback-models"}:
        return None
    try:
        aliases = list(load_model_registry(path).models)
    except ValueError:
        return []
    aliases.insert(0, "auto")
    selected, comma, fragment = value.rpartition(",")
    lead = f"{selected}," if comma else ""
    base = text[: -len(prefix)] if prefix else text
    return [
        (f"{base}{key}={lead}{alias}", "Model alias")
        for alias in aliases
        if alias.startswith(fragment) and alias not in selected.split(",")
    ]


def model_registry_summary(
    path: Path,
    selected: tuple[str, ...],
    selection: str = "",
) -> str:
    """Return a compact, secret-free chat model menu."""

    registry = load_model_registry(path)
    code = selected or registry.default_aliases("code_models")
    candidates = tuple(dict.fromkeys((*registry.default_aliases("code_models"), *code)))
    available = " · ".join(
        f"{alias} ({len(spec.endpoints_for('code'))} route(s))"
        for alias in candidates
        for spec in (registry.models[alias],)
    )
    return (
        f"Chat model · {', '.join(code)} · Policy "
        f"{selection or registry.defaults.get('selection', 'auto')}\n"
        f"Available · {available}\n"
        "Use /models <alias[,alias]> [policy=auto|spread|fixed]\n"
        f"{model_config_hint(path)}"
    )


def model_registry_brief(
    path: Path,
    selected: tuple[str, ...],
    selection: str = "",
) -> str:
    """Return a compact, secret-free startup summary for the TUI."""

    registry = load_model_registry(path)
    code = selected or registry.default_aliases("code_models")
    policy = str(selection or registry.defaults.get("selection") or "auto")
    return (
        f"Chat model · {', '.join(code)} · Policy {policy} · /models to switch"
    )


def parse_chat_model_selection(
    text: str,
    path: Path,
    *,
    current_policy: str,
) -> tuple[tuple[str, ...], str]:
    """Parse the lightweight ``/models`` chat-only selection syntax."""

    _, _, argument = str(text).strip().partition(" ")
    if not argument:
        raise ValueError("Choose at least one chat model alias.")
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        raise ValueError(f"Invalid model selection: {exc}") from exc
    policy = current_policy
    alias_tokens = []
    for token in tokens:
        if token.startswith("policy="):
            policy = token.partition("=")[2].strip().casefold()
        elif "=" in token:
            raise ValueError(f"Unknown model option: {token.partition('=')[0]}")
        else:
            alias_tokens.extend(token.split(","))
    registry = load_model_registry(path)
    candidates = registry.default_aliases("code_models")
    if len(alias_tokens) == 1 and alias_tokens[0].isdigit():
        index = int(alias_tokens[0]) - 1
        if not 0 <= index < len(candidates):
            raise ValueError(f"Model index must be between 1 and {len(candidates)}.")
        alias_tokens = [candidates[index]]
    if len(alias_tokens) == 1 and alias_tokens[0].casefold() == "auto":
        aliases = candidates
        policy = "auto"
    else:
        aliases = tuple(alias.strip() for alias in alias_tokens if alias.strip())
    if policy not in {"auto", "spread", "fixed"}:
        raise ValueError("Model policy must be auto, spread, or fixed.")
    unsupported = [alias for alias in aliases if alias not in candidates]
    if unsupported:
        raise ValueError(
            "Chat model alias must be one of: " + ", ".join(candidates)
        )
    registry.resolve(aliases, role="code")
    return aliases, policy


def research_model_question(path: Path, draft: LongResearchDraft) -> str:
    """Describe the optional per-run role choice without opening another panel."""

    registry = load_model_registry(path)
    available = ", ".join(registry.models)
    code = draft.code_models or list(registry.default_aliases("code_models"))
    feedback = draft.feedback_models or list(
        registry.default_aliases("feedback_models")
    )
    fixed_code = code[0] if code else "model-alias"
    fixed_feedback = feedback[0] if feedback else "model-alias"
    return (
        f"Research models · available: {available}\n"
        f"Default · Code {','.join(code)} · "
        f"Feedback {','.join(feedback)} · "
        f"Policy {draft.model_selection}\n"
        "Enter defaults, or models=<alias[,alias]> "
        "feedback-models=<alias[,alias]> model-policy=<auto|fixed>.\n"
        f"Fixed example · models={fixed_code} "
        f"feedback-models={fixed_feedback} model-policy=fixed"
    )


def research_model_overrides(text: str) -> dict[str, object]:
    """Read only explicit model options from a natural-language task command."""

    aliases = {
        "models": "code_models",
        "code-models": "code_models",
        "code_models": "code_models",
        "feedback-models": "feedback_models",
        "feedback_models": "feedback_models",
        "model-policy": "model_selection",
        "model_policy": "model_selection",
    }
    try:
        tokens = shlex.split(str(text).replace("，", " ").replace("；", " "))
    except ValueError:
        return {}
    values = {}
    for token in tokens:
        key, separator, raw = token.partition("=")
        field = aliases.get(key.casefold()) if separator else None
        if field is not None:
            values[field] = parse_field(field, raw)
    return values


def apply_research_model_choice(
    path: Path,
    draft: LongResearchDraft,
    text: str,
) -> None:
    """Apply one explicit per-run role answer to an onboarding draft."""

    registry = load_model_registry(path)
    if str(text).strip().casefold() in {"default", "defaults", "keep", "默认"}:
        code = draft.code_models or list(registry.default_aliases("code_models"))
        feedback = draft.feedback_models or list(
            registry.default_aliases("feedback_models")
        )
        registry.resolve(code, role="code")
        registry.resolve(feedback, role="feedback")
        draft.code_models = list(code)
        draft.feedback_models = list(feedback)
        draft.model_selection = str(
            draft.model_selection
            or registry.defaults.get("selection")
            or "auto"
        )
        return
    values = parse_constraints(text)
    allowed = {"code_models", "feedback_models", "model_selection"}
    unexpected = set(values) - allowed
    if unexpected or not values:
        raise ValueError(
            "Use defaults, or provide models=, feedback-models=, and model-policy=."
        )
    code = values.get("code_models", draft.code_models)
    feedback = values.get("feedback_models", draft.feedback_models)
    code = list(code or registry.default_aliases("code_models"))
    feedback = list(feedback or registry.default_aliases("feedback_models"))
    registry.resolve(code, role="code")
    registry.resolve(feedback, role="feedback")
    draft.code_models = code
    draft.feedback_models = feedback
    draft.model_selection = str(values.get("model_selection", draft.model_selection))


def configure_draft_models(draft: LongResearchDraft, path: Path) -> None:
    if not path.is_file():
        if draft.code_models or draft.feedback_models:
            raise ValueError(f"Model configuration does not exist: {path}")
        return
    registry = load_model_registry(path)
    code = draft.code_models or list(registry.default_aliases("code_models"))
    feedback = draft.feedback_models or list(registry.default_aliases("feedback_models"))
    if not code or not feedback:
        missing = "code_models" if not code else "feedback_models"
        raise ValueError(
            f"Model configuration defaults.{missing} must select at least one alias"
        )
    registry.resolve(code, role="code")
    registry.resolve(feedback, role="feedback")
    draft.code_models = list(code)
    draft.feedback_models = list(feedback)
    draft.model_selection = str(
        draft.model_selection or registry.defaults.get("selection") or "auto"
    )
    draft.model_config_path = str(path)


__all__ = [
    "apply_research_model_choice",
    "configure_draft_models",
    "model_alias_completions",
    "model_config_hint",
    "model_config_setup_hint",
    "model_registry_brief",
    "model_registry_summary",
    "parse_chat_model_selection",
    "research_model_overrides",
    "research_model_question",
]
