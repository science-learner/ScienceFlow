# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Stable REPL context and prompt projections."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from scienceflow.agent.core.runtime.run_policy import AutoContinuePolicy
from scienceflow.research.state.workspace.storage.git import normalize_workspace_git_track_globs

def _repl_environment_context_prompt(workspace_dir: str | Path) -> str:
    """Stable REPL environment context, generated once at session start."""
    shell = (os.environ.get("SHELL") or "bash").rsplit("/", 1)[-1] or "bash"
    current_date = datetime.now(timezone.utc).date().isoformat()
    return (
        "<environment_context>\n"
        f"  <cwd>{Path(workspace_dir).resolve()}</cwd>\n"
        f"  <shell>{shell}</shell>\n"
        f"  <current_date>{current_date}</current_date>\n"
        "  <timezone>Etc/UTC</timezone>\n"
        "</environment_context>"
    )


def _plain_repl_auto_continue_session(run_policy: object, teleport_mode: str) -> bool:
    """True for plain REPL code-agent sessions that should mimic generic CLI context."""
    return isinstance(run_policy, AutoContinuePolicy) and str(teleport_mode or "off") in ("", "off")


def _repl_code_agent_append_prompt() -> str:
    """Stable REPL-only system suffix aligned with a generic code agent."""
    return (
        "\n\n## Interactive REPL Code Agent\n"
        "You are in one continuous workspace session. Inspect files, modify files, "
        "run commands, validate outcomes, and continue from the current state.\n"
        "- Treat the fixed user task as authoritative; do not rewrite, summarize, "
        "or replace it.\n"
        "- Prefer small reversible changes with direct validation when changing "
        "the workspace.\n"
        "- For existing files, prefer targeted shell-level changes over "
        "whole-file rewrites. Use whole-file write mainly for new files or a "
        "clearly small replacement.\n"
        "- If you need exact code anchors, use targeted reads with offsets or "
        "limits instead of repeatedly reading full files.\n"
        "- Use workspace files and recorded outcomes as durable memory instead "
        "of relying on long conversation history.\n"
        "- For greetings, thanks, small talk, or questions answerable without "
        "workspace facts, reply naturally without tools.\n"
        "- Do not introduce planning systems, stage machines, or multi-agent "
        "search in REPL mode.\n"
    )


def _repl_bash_file_write_prompt() -> str:
    """REPL-only tool contract when file writes are performed through bash."""
    return (
        "\n\n## REPL Bash File Writes\n"
        "Available tools in this REPL session are `bash`, `read`, `grep`, `glob`, "
        "and `ls`. Use `bash` as the file-change tool.\n"
        "Create and modify files with `bash` so the complete content or exact "
        "rewrite operation remains in the tool-call arguments.\n"
        "- Preserve useful working artifacts before risky changes when practical.\n"
        "- For small existing-file changes, read exact anchors when needed and make "
        "one targeted shell-level change.\n"
        "- After a successful file-modifying bash command, validate with checks or "
        "runs instead of re-reading only to confirm the write.\n"
        "- Use `ls` / `glob` for file discovery, `grep` for text search, and `read` "
        "for exact file content. Use bash for shell execution, installs, validation "
        "runs, artifact preservation, and file changes.\n"
        "- A bash observation may be summarized, deduplicated, or truncated. Do not "
        "treat a compact bash result as complete ground truth when exact details matter.\n"
        "- If bash output was compacted or too broad, recover exact facts with a "
        "narrower command, `grep`, or paged `read` calls against a workspace file.\n"
        "- For verbose or long-running commands, write the full log to a workspace "
        "file and print only key metrics, exit status, artifact paths, and a compact tail.\n"
        "- Do not run verbose training or validation as `command | tail` when that "
        "is the only copy of the output. Save the full output to a workspace log "
        "first, then print extracted metrics and a compact tail.\n"
        "- During iterative optimization, record the current best metric, "
        "command/configuration, and artifact path in a workspace ledger. Record "
        "each substantive attempt, including failed attempts that change direction.\n"
        "- After a training/validation command returns, update that ledger before "
        "starting another risky experiment. If the metric improves, preserve the "
        "matching artifact(s) as the best-known candidate immediately. If a better "
        "training run has not produced the final artifact yet, run the matching "
        "prediction/export command and preserve that output first.\n"
        "- Before finishing a timed optimization run, promote the best known "
        "artifact to the expected final artifact path and align any metric file "
        "with that promoted artifact.\n"
        "- Keep bash output compact; the command arguments carry the durable file "
        "content.\n"
    )


def _append_repl_bash_file_write_prompt(text: str) -> str:
    block = _repl_bash_file_write_prompt()
    marker = "## REPL Bash File Writes"
    if marker in (text or ""):
        return text or ""
    return (text or "").rstrip() + block


def _repl_code_organization_prompt(hint: str | None) -> str:
    """Return an optional REPL user-side code organization hint."""
    raw = str(hint or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in {"0", "false", "no", "none", "off", "disabled"}:
        return ""
    if key in {
        "opt_solver",
        "optimization_solver",
        "artifact_solver",
        "generic_evaluator",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer solver-oriented workspace files for optimization tasks. "
            "A single `solution.py` is acceptable for a tiny baseline, but when "
            "the implementation grows, keep parsing, solver/search logic, "
            "validation, and artifact writing in clear functions or modules.\n\n"
            "Use the task's configured runtime and evaluator contract as the "
            "authority. Write the configured candidate artifact path exactly, "
            "and do not invent ML submission files or self-scored benchmark "
            "contracts unless the task description explicitly requests them.\n\n"
            "When iteration is expensive, preserve the best known candidate "
            "artifact before risky experiments, keep lightweight validation "
            "probes reproducible, and make resume state depend on workspace files "
            "rather than hidden process memory."
        )
    if key in {
        "beyond_mfiles",
        "beyond_multifile",
        "reusable_predict",
        "train_predict_boundary",
        "predict_reuse",
        "artifact_reuse",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer a reusable train/predict boundary over a fixed file count. "
            "A single `solution.py` is acceptable for a tiny or cheap baseline, "
            "but keep clear functions for loading data, preprocessing or feature "
            "extraction, training, prediction, and writing `submission.csv`.\n\n"
            "When training is expensive, artifacts are reusable, or the task is "
            "CV/NLP/audio with costly model fitting, split the workflow into "
            "runnable entrypoints:\n"
            "- `train.py`: train or tune once, save checkpoints/weights and every "
            "artifact required for inference, including tokenizer/vectorizer, "
            "label encoders, scalers, feature columns, thresholds, configs, and "
            "validation metrics.\n"
            "- `predict.py`: load saved artifacts and regenerate the root-level "
            "`submission.csv` without retraining. Keep this path deterministic "
            "and fast enough for repeated submit/postprocess checks.\n"
            "- `util.py` or shared functions: own preprocessing, feature extraction, "
            "metrics, path helpers, and submission schema validation so train and "
            "predict use identical transforms.\n\n"
            "After a strong model exists, prefer low-cost predict-only iterations "
            "before retraining: threshold tuning, calibration, postprocessing, "
            "test-time augmentation/inference settings, batch-size fixes, artifact "
            "loading checks, and submission formatting validation. Retrain only "
            "when the expected gain justifies the remaining budget.\n"
            "Do not duplicate train/test feature logic. If a transform changes, "
            "update the shared function or `util.py` first and keep train and "
            "predict aligned through that shared path."
        )
    if key in {
        "mfiles",
        "mle_mfiles",
        "mle_multifile",
        "multi_file",
        "multifile",
        "train_predict_submit_util",
        "train_predict_util",
    }:
        return (
            "## Code organization preference\n\n"
            "Prefer a small multi-file ML solution layout when it fits the task:\n"
            "- `util.py`: own shared data loading and feature engineering. Put the "
            "`build_features` / transform functions, metrics, seeds, and path helpers "
            "here so training and prediction use one identical feature pipeline.\n"
            "- `train.py`: train or tune models, record validation metrics, and save "
            "reusable artifacts by importing feature helpers from `util.py`.\n"
            "- `predict.py`: load saved artifacts, generate predictions deterministically, "
            "reuse `util.py` feature helpers, and write the required `submission.csv`.\n\n"
            "Do not duplicate feature-engineering code separately in training and "
            "prediction files. If a feature changes, update `util.py` first and keep "
            "both entrypoints aligned through that shared function.\n"
            "Keep entrypoints runnable from the shell and keep interfaces simple. "
            "Only add an extra submit/wrapper file if the task or runner genuinely "
            "needs a separate command; otherwise let `predict.py` be the submission "
            "entrypoint. If a single runnable entrypoint is required, provide a thin "
            "compatible wrapper that calls the multi-file pipeline. "
            "Do not over-split a tiny baseline when a simpler layout is clearly "
            "more reliable under the time budget."
        )
    return "## Code organization preference\n\n" + raw


def _repl_workspace_git_prompt(
    enabled: bool,
    track_globs: list[str] | tuple[str, ...] | None = None,
    *,
    auto_review: bool = False,
    auto_checkpoint: bool = True,
) -> str:
    """Return a short user-side hint for workspace-local source control."""
    if not enabled:
        return ""
    globs = ", ".join(f"`{p}`" for p in normalize_workspace_git_track_globs(track_globs))
    mode = (
        "Source checkpoints and submission snapshots are created automatically."
        if auto_checkpoint
        else "Source checkpoints are available in this local repository."
    )
    text = (
        "## Workspace source checkpoints\n\n"
        f"{mode} The repository tracks source/docs only ({globs}, plus "
        "`.gitignore`); datasets, model weights, logs, and submissions are not "
        "tracked by git.\n"
        "You may inspect checkpoints with `git log --oneline -- '*.py' '*.md'`, "
        "`git status --short`, and `git diff -- '*.py' '*.md'`. If an experiment "
        "breaks the source, you may restore tracked source files with "
        "`git restore --source=<commit> -- '*.py' '*.md'`. Do not create commits "
        "manually; keep focusing on modeling and validation."
    )
    if not auto_review:
        return text
    return (
        text
        + "\n"
        "Auto-review is enabled: before finalizing after a worse or broken "
        "experiment, inspect `git diff` and restore the best tracked source "
        "checkpoint if needed. Do not commit manually."
    )



__all__ = [
    "_append_repl_bash_file_write_prompt",
    "_plain_repl_auto_continue_session",
    "_repl_bash_file_write_prompt",
    "_repl_code_agent_append_prompt",
    "_repl_code_organization_prompt",
    "_repl_environment_context_prompt",
    "_repl_workspace_git_prompt",
]
