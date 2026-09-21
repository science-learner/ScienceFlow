# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

# Ruff: imports below the jieba warning filter are intentionally delayed.
# ruff: noqa: E402

import json
import os
import signal
import time
import warnings

# Suppress jieba SyntaxWarning (Python 3.13 compatibility) before any imports
# that might trigger jieba loading (e.g., via InquiryCraft dependencies).
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module="jieba",
)

from pathlib import Path
from typing import Any

import click

from scienceflow.foundation.config.runtime.environment import bootstrap_dotenv
from scienceflow.foundation.config.schema.settings import (
    Config,
    apply_repl_manifest_defaults,
)
from scienceflow.research.state.workspace.storage.git import (
    normalize_workspace_git_track_globs,
)

_SCIENCEFLOW_REPO_ROOT = Path(__file__).resolve().parents[3]


def _apply_task_workspace(cfg: Config, workspace: str | Path) -> None:
    cfg.task_workspace_root_dir = Path(workspace).expanduser().resolve(strict=False)


def _apply_parallel_manifest_cfg_env(cfg: Config) -> None:
    """Merge ``SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON`` from :class:`~scienceflow.runtime.parallel.execution.runner.ParallelRunner`."""
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if isinstance(data, dict):
        apply_repl_manifest_defaults(cfg, data)


def _bootstrap_env(env_file: Path | None = None) -> None:
    """Load user/project configuration and normalize endpoint lists."""
    bootstrap_dotenv(repo_root=_SCIENCEFLOW_REPO_ROOT, env_file=env_file)


def _is_repl_exit_line(task: str) -> bool:
    """True for exit / quit and common variants (e.g. ``exit()`` from Python habit)."""
    t = task.strip().lower().rstrip(";").strip()
    if not t:
        return False
    if t.endswith("()"):
        t = t[:-2].strip()
    return t in ("exit", "quit", "q")


def _repl_apply_dataset_symlinks(cfg: Config) -> None:
    """Symlink cfg.input_data_dir into workspace/dataset/ (same as run/prep fast paths)."""
    from scienceflow.research.solver.lnr.lifecycle.workspace.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    inp = Path(cfg.input_data_dir).expanduser().resolve(strict=False)
    if not str(inp).strip():
        return
    ws_dataset = Path(cfg.workspace_dir) / "dataset"
    if not ws_dataset.exists():
        source, _layout = resolve_workspace_dataset_source(inp)
        prepare_workspace_dataset_flat(source, ws_dataset)
    _repl_allow_input_data_root(cfg)


def _repl_allow_input_data_root(cfg: Config) -> Path | None:
    """Allow the code agent to read cfg.input_data_dir without exposing it as output."""
    inp = Path(cfg.input_data_dir).expanduser().resolve(strict=False)
    if not str(inp).strip():
        return None
    seen_pg = {
        Path(p).expanduser().resolve(strict=False)
        for p in (cfg.path_guard_extra_roots or [])
        if str(p).strip()
    }
    ir = inp.resolve()
    if ir not in seen_pg:
        cfg.path_guard_extra_roots = list(cfg.path_guard_extra_roots or [])
        cfg.path_guard_extra_roots.append(ir)
    return ir


def _repl_resolve_task_text(cfg: Config, *, manifest_task_raw: Any = None) -> tuple[str, str]:
    """Return (task_text, source) where source is 'file', 'resolved', or ''."""
    inner = Path(cfg.workspace_dir)
    for name in ("description.md", "task_desc.txt"):
        td = inner / name
        if td.is_file() or td.is_symlink():
            t = td.read_text(encoding="utf-8").strip()
            if t:
                return t, "file"
    from scienceflow.runtime.parallel.execution.runner import (
        _resolve_parallel_task_text,
    )

    exp = (cfg.exp_id or "").strip()
    if not exp:
        return "", ""
    try:
        t = _resolve_parallel_task_text(exp, manifest_task_raw).strip()
        if t:
            return t, "resolved"
    except ValueError:
        pass
    return "", ""


def _repl_materialize_task_desc(cfg: Config, text: str, source: str) -> None:
    if not text or source == "file":
        return
    out = Path(cfg.workspace_dir) / "description.md"
    if not out.exists():
        out.write_text(text, encoding="utf-8")


def _truthy_yaml(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _manifest_bool(
    defaults: dict[str, Any],
    task: dict[str, Any],
    key: str,
    fallback: bool = False,
) -> bool:
    for block in (task, defaults):
        if not isinstance(block, dict) or key not in block:
            continue
        parsed = _truthy_yaml(block.get(key))
        if parsed is not None:
            return parsed
    return fallback


def _manifest_str(defaults: dict[str, Any], task: dict[str, Any], key: str) -> str:
    for block in (task, defaults):
        if not isinstance(block, dict) or key not in block:
            continue
        value = block.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _repl_normalized_profile(value: Any) -> str:
    raw = str(value or "lite").strip().lower().replace("-", "_")
    aliases = {
        "lite": "lite",
        "legacy": "legacy",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(set(aliases)))
        raise click.ClickException(f"Invalid repl_profile {value!r}; expected one of: {allowed}")
    return aliases[raw]


def _repl_normalized_tool_preset(value: Any, *, profile: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return "write_edit" if profile == "legacy" else "bash_write"
    aliases = {
        "bash": "bash_write",
        "bash_only": "bash_write",
        "bash_write": "bash_write",
        "write": "write_edit",
        "write_edit": "write_edit",
        "legacy": "write_edit",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(set(aliases)))
        raise click.ClickException(
            f"Invalid repl_tool_preset {value!r}; expected one of: {allowed}",
        )
    return aliases[raw]


def _repl_positive_int(value: Any, *, fallback: int, key: str) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"{key} must be an integer, got {value!r}") from exc
    return parsed if parsed > 0 else int(fallback)


def _repl_resolve_runtime_options(
    cfg: Config,
    *,
    auto_first_user_enabled: bool,
) -> dict[str, Any]:
    """Resolve REPL-only runtime profile."""
    profile = _repl_normalized_profile(getattr(cfg, "repl_profile", "lite"))
    tool_preset = _repl_normalized_tool_preset(
        getattr(cfg, "repl_tool_preset", ""),
        profile=profile,
    )
    max_steps = _repl_positive_int(
        getattr(cfg, "repl_max_steps", 0),
        fallback=int(getattr(cfg, "qa_max_steps", 20) or 20),
        key="repl_max_steps",
    )
    bash_max_output_chars = _repl_positive_int(
        getattr(cfg, "repl_bash_max_output_chars", 0),
        fallback=8000,
        key="repl_bash_max_output_chars",
    )
    bash_max_stream_line_chars = _repl_positive_int(
        getattr(cfg, "repl_bash_max_stream_line_chars", 0),
        fallback=2400,
        key="repl_bash_max_stream_line_chars",
    )
    bash_observation_summary = bool(
        getattr(cfg, "repl_bash_observation_summary", profile != "legacy"),
    )
    stable_system_prompt = bool(
        getattr(cfg, "repl_stable_system_prompt", profile != "legacy"),
    )
    pin_environment_context = bool(
        getattr(cfg, "repl_pin_environment_context", profile != "legacy"),
    )
    code_organization_hint = str(
        getattr(cfg, "repl_code_organization_hint", "") or "",
    ).strip()
    workspace_git_enabled = bool(
        getattr(cfg, "repl_workspace_git_enabled", profile != "legacy"),
    )
    if profile == "legacy":
        workspace_git_enabled = False
    workspace_git_track_globs = normalize_workspace_git_track_globs(
        getattr(cfg, "repl_workspace_git_track_globs", None),
    )
    workspace_git_auto_review = bool(
        getattr(cfg, "repl_workspace_git_auto_review", False),
    )
    workspace_git_auto_checkpoint = bool(
        getattr(cfg, "repl_workspace_git_auto_checkpoint", workspace_git_enabled),
    )
    if not workspace_git_enabled:
        workspace_git_auto_checkpoint = False
    pin_task_when_auto = bool(
        getattr(cfg, "repl_pin_task_description_when_auto_first_user", False),
    )
    return {
        "profile": profile,
        "tool_preset": tool_preset,
        "max_steps": max_steps,
        "stable_system_prompt": stable_system_prompt,
        "pin_environment_context": pin_environment_context,
        "code_organization_hint": code_organization_hint,
        "workspace_git_enabled": workspace_git_enabled,
        "workspace_git_track_globs": list(workspace_git_track_globs),
        "workspace_git_auto_review": workspace_git_auto_review,
        "workspace_git_auto_checkpoint": workspace_git_auto_checkpoint,
        "pin_task_description": (not auto_first_user_enabled) or pin_task_when_auto,
        "repl_bash_write_mode": tool_preset == "bash_write",
        "bash_max_output_chars": bash_max_output_chars,
        "bash_max_stream_line_chars": bash_max_stream_line_chars,
        "bash_observation_summary": bash_observation_summary,
    }


def _read_manifest_text_file(path_value: str, *, manifest_dir: Path | None) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if manifest_dir is not None:
            candidates.append(manifest_dir / path)
        candidates.append(_SCIENCEFLOW_REPO_ROOT / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    tried = ", ".join(str(p) for p in candidates)
    raise click.ClickException(f"Manifest text file not found for {raw!r}; tried: {tried}")


def _repl_build_first_user_query(task_text: str) -> str:
    """Build the automatic first REPL turn without legacy solver prompt dependencies."""
    body = (task_text or "").strip()
    if body:
        return (
            "Design and implement a strong solution for this task. Work inside the current "
            "workspace, inspect the available dataset under `dataset/`, create the necessary "
            "source files, run validation, and produce the final submission artifact.\n\n"
            "## Task\n\n"
            f"{body}"
        )
    return (
        "Design and implement a strong solution for the task available in this workspace. "
        "Inspect `dataset/`, create the necessary source files, run validation, and produce "
        "the final submission artifact."
    )


def _install_cleanup_signals() -> None:
    """Register SIGINT/SIGTERM/SIGHUP handlers that sweep all live child PGIDs.

    Called once at CLI entry.  The handler sends SIGTERM to every process
    group in the live registry, waits 5 s, then sends SIGKILL to survivors,
    and finally reinstates the default handler so the process exits normally.
    """
    from scienceflow.runtime.core.process.utils import kill_all_live_pgids

    def _handler(signum: int, frame: object) -> None:
        # SIGTERM first pass
        kill_all_live_pgids(sig=signal.SIGTERM)
        # Give processes a chance to clean up (DDP destroy_process_group etc.)
        time.sleep(5)
        # SIGKILL survivors
        kill_all_live_pgids(sig=signal.SIGKILL)
        # Restore default and re-raise so the process exits with correct status
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass  # SIGHUP unavailable on some platforms


@click.group()
@click.option("--env-file", type=click.Path(path_type=Path, dir_okay=False), default=None,
              envvar="SCIENCEFLOW_ENV_FILE", help="Explicit dotenv configuration file.")
@click.pass_context
def main(ctx, env_file):
    """ScienceFlow - Autonomous ML Agent Framework"""
    if ctx.invoked_subcommand == "config":
        if env_file is not None:
            os.environ["SCIENCEFLOW_ENV_FILE"] = str(env_file)
        return
    try:
        _bootstrap_env(env_file)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    _install_cleanup_signals()



# Command implementations live in focused modules; this package keeps only the
# shared CLI context and the stable public ``main`` entry point.
from scienceflow.interfaces.cli.commands.inspect.monitor import (
    monitor_cmd,
    monitor_trace_cmd,
)
from scienceflow.interfaces.cli.commands.inspect.replay import replay_prepare_cmd
from scienceflow.interfaces.cli.commands.inspect.resources import resource_summary_cmd
from scienceflow.interfaces.cli.commands.inspect.web import web
from scienceflow.interfaces.cli.commands.run.control import resume, status, stop
from scienceflow.interfaces.cli.commands.run.parallel import parallel
from scienceflow.interfaces.cli.commands.run.repl import repl
from scienceflow.interfaces.cli.commands.run.task import prep, run
from scienceflow.interfaces.cli.commands.run.tui import tui

for _command in (
    prep,
    run,
    stop,
    resume,
    status,
    repl,
    parallel,
    tui,
    monitor_cmd,
    monitor_trace_cmd,
    web,
    resource_summary_cmd,
    replay_prepare_cmd,
):
    main.add_command(_command)


# Additive composition point: existing ScienceFlow commands keep their exact names and
# handlers, while ``scienceflow agent ...`` exposes the independent InquiryCraft CLI using
# ScienceFlow supplies its structured model registry while InquiryCraft owns the CLI.
from inspect import signature as _call_signature

from inquirycraft.cli import create_cli as _create_inquirycraft_cli

from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory, plural_cli_options

_inquirycraft_cli_options: dict[str, Any] = {
    "name": "agent",
    "runtime_factory": PoolRuntimeFactory(),
    "model_config_default": Path("~/.config/scienceflow/models.json"),
    "model_default_key": "code_models",
}
if "include_tui" in _call_signature(_create_inquirycraft_cli).parameters:
    _inquirycraft_cli_options["include_tui"] = False

main.add_command(
    plural_cli_options(_create_inquirycraft_cli(**_inquirycraft_cli_options)),
    name="agent",
)


from scienceflow.interfaces.cli.user_config import (
    add_environment_options,
    config_command,
)

main.add_command(config_command)
add_environment_options(main)

if __name__ == "__main__":
    main()
